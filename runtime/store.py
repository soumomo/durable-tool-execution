from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import sqlite3
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Any, Self, TypeVar
from uuid import uuid4

from .models import (
    BackpressureError,
    IdempotencyConflictError,
    JobEvent,
    JobRecord,
    JobStatus,
    JSONValue,
    RuntimeConfig,
    StaleLeaseError,
    ToolPolicy,
)

T = TypeVar("T")
OUTSTANDING = "('PENDING', 'LEASED', 'RETRY_WAIT')"
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'PENDING', 'LEASED', 'RETRY_WAIT', 'SUCCEEDED', 'FAILED', 'UNKNOWN_OUTCOME'
    )),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    idempotent INTEGER NOT NULL CHECK (idempotent IN (0, 1)),
    timeout_ms INTEGER NOT NULL CHECK (timeout_ms > 0),
    max_retries INTEGER NOT NULL CHECK (max_retries >= 0),
    base_backoff_ms INTEGER NOT NULL CHECK (base_backoff_ms >= 0),
    lease_owner TEXT,
    lease_expires_at REAL,
    next_eligible_at REAL NOT NULL,
    result_json TEXT,
    error_text TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    CHECK (
        (status = 'LEASED' AND lease_owner IS NOT NULL
         AND lease_expires_at IS NOT NULL AND attempt_count > 0)
        OR (status != 'LEASED' AND lease_owner IS NULL AND lease_expires_at IS NULL)
    ),
    CHECK (
        (status IN ('SUCCEEDED', 'FAILED', 'UNKNOWN_OUTCOME') AND completed_at IS NOT NULL)
        OR (status IN ('PENDING', 'LEASED', 'RETRY_WAIT') AND completed_at IS NULL)
    ),
    CHECK ((status = 'SUCCEEDED') = (result_json IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    attempt INTEGER NOT NULL CHECK (attempt >= 0),
    event_type TEXT NOT NULL CHECK (event_type IN (
        'STATE_TRANSITION', 'STALE_COMPLETION_FENCED'
    )),
    from_status TEXT,
    to_status TEXT NOT NULL,
    timestamp REAL NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS jobs_eligible
    ON jobs(next_eligible_at, created_at, job_id)
    WHERE status IN ('PENDING', 'RETRY_WAIT');
CREATE INDEX IF NOT EXISTS jobs_expired
    ON jobs(lease_expires_at) WHERE status = 'LEASED';
CREATE INDEX IF NOT EXISTS jobs_outstanding
    ON jobs(status) WHERE status IN ('PENDING', 'LEASED', 'RETRY_WAIT');
CREATE INDEX IF NOT EXISTS events_by_job ON job_events(job_id, event_id);
"""


def _validate_json(value: Any, ancestors: set[int]) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if type(value) not in (dict, list):
        raise TypeError(f"Not a JSON value: {type(value).__name__}")
    if id(value) in ancestors:
        raise ValueError("Circular JSON value")
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise TypeError("JSON object keys must be strings")
            children = value.values()
        else:
            children = value
        for child in children:
            _validate_json(child, ancestors)
    finally:
        ancestors.remove(id(value))


def _canonical_json(value: Any) -> str:
    try:
        _validate_json(value, set())
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except RecursionError as exc:
        raise ValueError("JSON value is nested too deeply") from exc


def _timestamp(now: float | None) -> float:
    if now is None:
        return time.time()
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError("now must be a finite Unix timestamp")
    return float(now)


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


async def _finish_cleanup(future: asyncio.Future[T]) -> T:
    # Cleanup must reach the owning thread even if shutdown cancels its caller again.
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(future)
            break
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class SQLiteJobStore:
    """One connection, owned by one executor thread; every mutation is transactional.

    SQLite is authoritative. Cancellation of a caller does not prove its queued
    transaction was aborted; retry admission with the same idempotency key.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self._random = random_source
        self._connection: sqlite3.Connection | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state = "new"
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            loop = asyncio.get_running_loop()
            if self._state == "open":
                if loop is not self._loop:
                    raise RuntimeError("Store belongs to a different event loop")
                return
            if self._state != "new":
                raise RuntimeError(
                    "A closed store cannot be restarted; create a new instance"
                )
            self._loop = loop
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="tool-runtime-db"
            )
            self._state = "starting"
            try:
                await loop.run_in_executor(self._executor, self._open)
            except BaseException:
                self._state = "closing"
                cleanup = loop.run_in_executor(self._executor, self._close)
                try:
                    await _finish_cleanup(cleanup)
                finally:
                    self._executor.shutdown(wait=False)
                    self._state = "closed"
                raise
            self._state = "open"

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._state == "closed":
                return
            self._state = "closing"
            if self._executor is None:
                self._state = "closed"
                return
            cleanup = asyncio.get_running_loop().run_in_executor(
                self._executor, self._close
            )
            try:
                await _finish_cleanup(cleanup)
            finally:
                self._executor.shutdown(wait=False)
                self._state = "closed"

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _run(self, function: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        if self._state != "open" or self._executor is None:
            raise RuntimeError("Store is not open; call start() before using it")
        if loop is not self._loop:
            raise RuntimeError("Store belongs to a different event loop")
        return await loop.run_in_executor(self._executor, partial(function, *args))

    def _open(self) -> None:
        self._connection = sqlite3.connect(self.config.db_path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        db = self._db
        db.execute("PRAGMA busy_timeout=5000")
        mode = db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode != "wal":
            raise RuntimeError(f"SQLite did not enable WAL mode: {mode}")
        db.execute(f"PRAGMA synchronous={self.config.synchronous_mode}")
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)

    def _close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Database connection is not open")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = self._db
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        policy = ToolPolicy(
            row["tool_name"],
            bool(row["idempotent"]),
            row["timeout_ms"],
            row["max_retries"],
            row["base_backoff_ms"],
        )
        fields = {
            field: row[field]
            for field in JobRecord.__dataclass_fields__
            if field not in ("policy", "status")
        }
        return JobRecord(**fields, policy=policy, status=JobStatus(row["status"]))

    def _get_row(self, job_id: str) -> sqlite3.Row:
        row = self._db.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def _event(
        self,
        job_id: str,
        attempt: int,
        from_status: str | None,
        to_status: str,
        now: float,
        detail: str | None = None,
        *,
        event_type: str = "STATE_TRANSITION",
    ) -> None:
        self._db.execute(
            "INSERT INTO job_events "
            "(job_id, attempt, event_type, from_status, to_status, timestamp, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, attempt, event_type, from_status, to_status, now, detail),
        )

    async def admit_job(
        self,
        policy: ToolPolicy,
        args: dict[str, JSONValue],
        idempotency_key: str,
        *,
        now: float | None = None,
    ) -> tuple[JobRecord, bool]:
        _nonempty(idempotency_key, "idempotency_key")
        if type(args) is not dict:
            raise TypeError("Tool arguments must be a JSON object")
        args_json = _canonical_json(args)
        identity = _canonical_json(policy.name) + ":" + args_json
        fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return await self._run(
            self._admit_job, policy, args_json, fingerprint, idempotency_key, now
        )

    def _admit_job(
        self,
        policy: ToolPolicy,
        args_json: str,
        fingerprint: str,
        key: str,
        now: float | None,
    ) -> tuple[JobRecord, bool]:
        with self._transaction() as db:
            existing = db.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["tool_name"] != policy.name
                    or existing["args_hash"] != fingerprint
                ):
                    raise IdempotencyConflictError(
                        f"Key {key!r} already identifies a different request"
                    )
                return self._record(existing), False
            if self._count_outstanding_jobs() >= self.config.max_outstanding_jobs:
                raise BackpressureError("Maximum number of unfinished jobs reached")
            when = _timestamp(now)
            job_id = uuid4().hex
            db.execute(
                "INSERT INTO jobs (job_id, idempotency_key, tool_name, args_json, args_hash, "
                "status, idempotent, timeout_ms, max_retries, base_backoff_ms, "
                "next_eligible_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    key,
                    policy.name,
                    args_json,
                    fingerprint,
                    int(policy.idempotent),
                    policy.timeout_ms,
                    policy.max_retries,
                    policy.base_backoff_ms,
                    when,
                    when,
                    when,
                ),
            )
            self._event(job_id, 0, None, "PENDING", when)
            return self._record(self._get_row(job_id)), True

    async def claim_next_job(
        self, worker_id: str, *, now: float | None = None
    ) -> JobRecord | None:
        _nonempty(worker_id, "worker_id")
        return await self._run(self._claim_next_job, worker_id, now)

    def _claim_next_job(self, worker_id: str, now: float | None) -> JobRecord | None:
        with self._transaction() as db:
            when = _timestamp(now)
            row = db.execute(
                "SELECT * FROM jobs WHERE status IN ('PENDING', 'RETRY_WAIT') "
                "AND next_eligible_at <= ? ORDER BY next_eligible_at, created_at, job_id LIMIT 1",
                (when,),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status = 'LEASED', attempt_count = attempt_count + 1, "
                "lease_owner = ?, lease_expires_at = ?, started_at = ?, updated_at = ? "
                "WHERE job_id = ? AND status = ?",
                (
                    worker_id,
                    when + self.config.lease_duration_ms / 1000,
                    when,
                    when,
                    row["job_id"],
                    row["status"],
                ),
            )
            self._event(
                row["job_id"], row["attempt_count"] + 1, row["status"], "LEASED", when
            )
            return self._record(self._get_row(row["job_id"]))

    async def heartbeat_lease(
        self, job_id: str, attempt_count: int, *, now: float | None = None
    ) -> bool:
        return await self._run(self._heartbeat_lease, job_id, attempt_count, now)

    def _heartbeat_lease(self, job_id: str, attempt: int, now: float | None) -> bool:
        with self._transaction() as db:
            when = _timestamp(now)
            cursor = db.execute(
                "UPDATE jobs SET lease_expires_at = ?, updated_at = ? "
                "WHERE job_id = ? AND status = 'LEASED' AND attempt_count = ? "
                "AND lease_expires_at > ?",
                (
                    when + self.config.lease_duration_ms / 1000,
                    when,
                    job_id,
                    attempt,
                    when,
                ),
            )
            return cursor.rowcount == 1

    async def complete_job(
        self,
        job_id: str,
        attempt_count: int,
        result: JSONValue,
        *,
        now: float | None = None,
    ) -> JobRecord:
        result_json = _canonical_json(result)
        return await self._run(
            self._write_outcome,
            job_id,
            attempt_count,
            result_json,
            None,
            False,
            False,
            now,
        )

    async def fail_or_retry_job(
        self,
        job_id: str,
        attempt_count: int,
        error_text: str,
        *,
        retryable: bool,
        outcome_unknown: bool = True,
        now: float | None = None,
    ) -> JobRecord:
        _nonempty(error_text, "error_text")
        return await self._run(
            self._write_outcome,
            job_id,
            attempt_count,
            None,
            error_text,
            retryable,
            outcome_unknown,
            now,
        )

    def _failure_status(
        self,
        row: sqlite3.Row,
        *,
        retryable: bool,
        outcome_unknown: bool,
    ) -> JobStatus:
        if not row["idempotent"] and outcome_unknown:
            return JobStatus.UNKNOWN_OUTCOME
        if (
            row["idempotent"]
            and retryable
            and row["attempt_count"] <= row["max_retries"]
        ):
            return JobStatus.RETRY_WAIT
        return JobStatus.FAILED

    def _retry_at(self, row: sqlite3.Row, now: float) -> float:
        # Even a 1 ms base reaches the 60 s cap by exponent 16.
        cap_ms = min(
            row["base_backoff_ms"] * 2 ** min(row["attempt_count"] - 1, 16), 60_000
        )
        fraction = self._random()
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError(
                "random_source must return a finite number between 0 and 1"
            )
        return now + cap_ms * fraction / 1000

    def _transition_outcome(
        self,
        row: sqlite3.Row,
        status: JobStatus,
        now: float,
        result_json: str | None,
        error_text: str | None,
        *,
        expired: bool = False,
    ) -> JobRecord:
        eligible_at = (
            self._retry_at(row, now)
            if status is JobStatus.RETRY_WAIT
            else row["next_eligible_at"]
        )
        expiry_comparison = "<=" if expired else ">"
        cursor = self._db.execute(
            "UPDATE jobs SET status = ?, result_json = ?, error_text = ?, updated_at = ?, "
            "completed_at = ?, next_eligible_at = ?, lease_owner = NULL, lease_expires_at = NULL "
            "WHERE job_id = ? AND status = 'LEASED' AND attempt_count = ? "
            f"AND lease_expires_at {expiry_comparison} ?",
            (
                status.value,
                result_json,
                error_text,
                now,
                now if status.terminal else None,
                eligible_at,
                row["job_id"],
                row["attempt_count"],
                now,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleLeaseError(f"Lease changed for job {row['job_id']}")
        self._event(
            row["job_id"], row["attempt_count"], "LEASED", status.value, now, error_text
        )
        return self._record(self._get_row(row["job_id"]))

    def _write_outcome(
        self,
        job_id: str,
        attempt: int,
        result_json: str | None,
        error_text: str | None,
        retryable: bool,
        outcome_unknown: bool,
        now: float | None,
    ) -> JobRecord:
        outcome = None
        with self._transaction():
            when = _timestamp(now)
            row = self._get_row(job_id)
            live = (
                row["status"] == "LEASED"
                and row["attempt_count"] == attempt
                and row["lease_expires_at"] > when
            )
            if not live:
                self._event(
                    job_id,
                    attempt,
                    row["status"],
                    row["status"],
                    when,
                    "Outcome discarded: attempt does not own a live lease",
                    event_type="STALE_COMPLETION_FENCED",
                )
            else:
                status = (
                    JobStatus.SUCCEEDED
                    if error_text is None
                    else self._failure_status(
                        row, retryable=retryable, outcome_unknown=outcome_unknown
                    )
                )
                outcome = self._transition_outcome(
                    row, status, when, result_json, error_text
                )
        # Commit the fencing audit event before surfacing the rejected write.
        if outcome is None:
            raise StaleLeaseError(
                f"Attempt {attempt} does not own a live lease for job {job_id}"
            )
        return outcome

    async def recover_expired_leases(
        self, *, now: float | None = None
    ) -> list[JobRecord]:
        return await self._run(self._recover_expired_leases, now)

    def _recover_expired_leases(self, now: float | None) -> list[JobRecord]:
        with self._transaction() as db:
            when = _timestamp(now)
            rows = db.execute(
                "SELECT * FROM jobs WHERE status = 'LEASED' AND lease_expires_at <= ? "
                "ORDER BY lease_expires_at, job_id",
                (when,),
            ).fetchall()
            recovered = []
            for row in rows:
                status = self._failure_status(row, retryable=True, outcome_unknown=True)
                error = (
                    "Worker lease expired during non-idempotent execution; manual reconciliation required"
                    if not row["idempotent"]
                    else "Worker lease expired during execution"
                )
                recovered.append(
                    self._transition_outcome(
                        row, status, when, None, error, expired=True
                    )
                )
            return recovered

    async def get_job(self, job_id: str) -> JobRecord:
        return await self._run(self._get_job, job_id)

    def _get_job(self, job_id: str) -> JobRecord:
        return self._record(self._get_row(job_id))

    async def get_events(self, job_id: str) -> list[JobEvent]:
        return await self._run(self._get_events, job_id)

    def _get_events(self, job_id: str) -> list[JobEvent]:
        rows = self._db.execute(
            "SELECT * FROM job_events WHERE job_id = ? ORDER BY event_id", (job_id,)
        ).fetchall()
        return [
            JobEvent(
                row["event_id"],
                row["job_id"],
                row["attempt"],
                row["event_type"],
                JobStatus(row["from_status"])
                if row["from_status"] is not None
                else None,
                JobStatus(row["to_status"]),
                row["timestamp"],
                row["detail"],
            )
            for row in rows
        ]

    async def list_jobs(self) -> list[JobRecord]:
        return await self._run(self._list_jobs)

    def _list_jobs(self) -> list[JobRecord]:
        rows = self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at, job_id"
        ).fetchall()
        return [self._record(row) for row in rows]

    async def list_unfinished_jobs(self) -> list[JobRecord]:
        return await self._run(self._list_unfinished_jobs)

    def _list_unfinished_jobs(self) -> list[JobRecord]:
        rows = self._db.execute(
            f"SELECT * FROM jobs WHERE status IN {OUTSTANDING} ORDER BY created_at, job_id"
        ).fetchall()
        return [self._record(row) for row in rows]

    async def count_outstanding_jobs(self) -> int:
        return await self._run(self._count_outstanding_jobs)

    def _count_outstanding_jobs(self) -> int:
        return self._db.execute(
            f"SELECT count(*) FROM jobs WHERE status IN {OUTSTANDING}"
        ).fetchone()[0]
