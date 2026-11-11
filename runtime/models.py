from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


class JobStatus(str, Enum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"

    @property
    def terminal(self) -> bool:
        return self in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.UNKNOWN_OUTCOME,
        }


def _integer(name: str, value: int, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True) #froezen = True: read-only
class ToolPolicy:
    name: str
    idempotent: bool
    timeout_ms: int = 2000
    max_retries: int = 3
    base_backoff_ms: int = 50

    def __post_init__(self) -> None: # __post_init__ : validate the inputs the moment someone creates the object
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name must be a nonempty string")
        if type(self.idempotent) is not bool:
            raise ValueError("idempotent must be a bool")
        _integer("timeout_ms", self.timeout_ms)
        _integer("max_retries", self.max_retries, minimum=0)
        _integer("base_backoff_ms", self.base_backoff_ms, minimum=0)

    @property #dynamic calculation
    def max_attempts(self) -> int:
        return 1 + self.max_retries


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    db_path: str
    max_workers: int = 8
    max_outstanding_jobs: int = 500
    lease_duration_ms: int = 1000
    heartbeat_interval_ms: int = 300
    synchronous_mode: Literal["FULL", "NORMAL"] = "FULL"
    poll_interval_ms: int = 50
    recovery_interval_ms: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.db_path, str) or not self.db_path.strip():
            raise ValueError("db_path must be a nonempty file path")
        if self.db_path == ":memory:":
            raise ValueError("Durable storage requires a file-backed database")
        for name in (
            "max_workers",
            "max_outstanding_jobs",
            "lease_duration_ms",
            "heartbeat_interval_ms",
            "poll_interval_ms",
            "recovery_interval_ms",
        ):
            _integer(name, getattr(self, name))
        if self.heartbeat_interval_ms >= self.lease_duration_ms:
            raise ValueError(
                "heartbeat_interval_ms must be less than lease_duration_ms"
            )
        if self.synchronous_mode not in ("FULL", "NORMAL"):
            raise ValueError("synchronous_mode must be FULL or NORMAL")


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    idempotency_key: str
    policy: ToolPolicy
    args_json: str
    args_hash: str
    status: JobStatus
    attempt_count: int
    lease_owner: str | None
    lease_expires_at: float | None
    next_eligible_at: float
    result_json: str | None
    error_text: str | None
    created_at: float
    updated_at: float
    started_at: float | None
    completed_at: float | None

    @property
    def tool_name(self) -> str:
        return self.policy.name

    @property
    def args(self) -> dict[str, JSONValue]:
        return json.loads(self.args_json)

    @property
    def result(self) -> JSONValue:
        return json.loads(self.result_json) if self.result_json is not None else None


@dataclass(frozen=True, slots=True)
class JobEvent:
    event_id: int
    job_id: str
    attempt: int
    event_type: str
    from_status: JobStatus | None
    to_status: JobStatus
    timestamp: float
    detail: str | None


class IdempotencyConflictError(ValueError):
    """A submission key already identifies a different tool or arguments."""


class BackpressureError(RuntimeError):
    """The durable unfinished-job limit has been reached."""


class StaleLeaseError(RuntimeError):
    """The attempt no longer owns a live lease and cannot write an outcome."""


class ToolRetryableError(RuntimeError):
    """A tool reports a transient failure; its policy still governs retries."""


class ToolFatalError(RuntimeError):
    """A tool reports a definitive failure rather than an ambiguous outcome."""
