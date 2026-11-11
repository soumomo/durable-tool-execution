"""Real SIGKILL/restart check; no API keys, third-party services, or user DBs.

``await run_chaos(synchronous_mode="FULL", timeout=30)`` returns a ChaosReport
with counts, measured elapsed time, ledger evidence, and individually verified
checks. The temporary runtime DB and independent mock-effect ledger are removed
on exit. This tests process crashes, not power loss or OS/filesystem failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from contextlib import closing, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from runtime.engine import DurableToolRuntime
from runtime.models import (
    JobStatus,
    JSONValue,
    RuntimeConfig,
    ToolFatalError,
    ToolPolicy,
    ToolRetryableError,
)
from runtime.registry import ToolRegistry

# The same immutable policies are registered in both processes. One retry means
# two total attempts, including a lease lost to SIGKILL.
_POLICIES = {
    kind: ToolPolicy(
        kind, idempotent, timeout_ms=3_600_000, max_retries=1, base_backoff_ms=20
    )
    for kind, idempotent in (
        ("fast", True),
        ("nonidem_blocker", False),
        ("idem_blocker", True),
        ("flaky", True),
        ("fatal", False),
        ("slow", True),
    )
}
_WORKLOAD = tuple(
    (f"{index:03d}:{kind}", kind)
    for index, kind in enumerate(
        ["fast"] * 40
        + ["nonidem_blocker"] * 4
        + ["idem_blocker"] * 4
        + ["flaky"] * 20
        + ["fatal"] * 8
        + ["slow", "fast"] * 12
    )
)
_TERMINAL = ("SUCCEEDED", "FAILED", "UNKNOWN_OUTCOME")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _counts(jobs: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["status"] for row in jobs.values())
    return {status.value: counts[status.value] for status in JobStatus}


@dataclass(frozen=True)
class ChaosReport:
    synchronous_mode: str
    accepted: int
    before_crash: dict[str, int]
    after_recovery: dict[str, int]
    killed_exit_code: int
    recovery_exit_code: int
    wal_present_after_sigkill: bool
    evidence: dict[str, int]
    checks: dict[str, bool]
    elapsed_seconds: float

    @property
    def unfinished(self) -> int:
        return sum(
            count
            for status, count in self.after_recovery.items()
            if status not in _TERMINAL
        )

    @property
    def conserved(self) -> bool:
        return (
            self.unfinished == 0
            and self.accepted
            == sum(self.after_recovery.get(status, 0) for status in _TERMINAL)
            and self.checks.get("accepted_ids_exactly_once", False)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "conserved": self.conserved,
            "unfinished": self.unfinished,
        }


def format_report(report: ChaosReport) -> str:
    """Plain, path/PID-free output suitable for pasting verbatim into the README."""
    counts, evidence = report.after_recovery, report.evidence
    passed = bool(report.checks) and all(report.checks.values()) and report.conserved
    return "\n".join(
        (
            f"SIGKILL chaos: SQLite synchronous={report.synchronous_mode}",
            (
                f"Crash point: accepted={report.accepted}, "
                f"SUCCEEDED={report.before_crash['SUCCEEDED']}, "
                f"LEASED={report.before_crash['LEASED']} (4 idempotent + 4 non-idempotent), "
                f"PENDING={report.before_crash['PENDING']}"
            ),
            f"Child exit: {report.killed_exit_code} (SIGKILL); recovery exit: {report.recovery_exit_code}",
            (
                f"Recovery: SUCCEEDED={counts['SUCCEEDED']}, FAILED={counts['FAILED']}, "
                f"UNKNOWN_OUTCOME={counts['UNKNOWN_OUTCOME']}, unfinished={report.unfinished}"
            ),
            (
                f"Conservation: {report.accepted} accepted = {counts['SUCCEEDED']} succeeded + "
                f"{counts['FAILED']} failed + {counts['UNKNOWN_OUTCOME']} unknown "
                f"({'PASS' if report.conserved else 'FAIL'})"
            ),
            (
                f"Precompleted: {evidence['precompleted_preserved']} unchanged; "
                f"{evidence['precompleted_reexecutions']} reexecutions"
            ),
            (
                f"Non-idempotent blockers: {evidence['nonidem_effects_before_crash']} effects before crash; "
                f"{evidence['unknown_reexecutions']} UNKNOWN_OUTCOME replays"
            ),
            (
                f"Idempotent blockers: {evidence['idem_blocker_calls']} invocations; "
                f"{evidence['idem_blocker_effects']} deduplicated effects"
            ),
            f"Fail-once tools: {evidence['flaky_calls']} invocations; {evidence['flaky_effects']} effects",
            (
                "SQLite runtime + ledger: "
                f"integrity_check={'ok' if report.checks['sqlite_integrity'] else 'FAIL'}; "
                f"foreign_key_check={'clean' if report.checks['sqlite_foreign_keys'] else 'FAIL'}"
            ),
            (
                "Post-SIGKILL WAL observed before reopening: "
                f"{'yes' if report.wal_present_after_sigkill else 'no'}"
            ),
            (
                f"Verification checks: {sum(report.checks.values())}/{len(report.checks)} "
                f"({'PASS' if passed else 'FAIL'})"
            ),
            f"Elapsed: {report.elapsed_seconds:.3f} s",
        )
    )


def _init_ledger(directory: Path) -> None:
    with closing(sqlite3.connect(directory / "effects.sqlite3")) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(
            "CREATE TABLE tool_keys (job_key TEXT PRIMARY KEY);"
            "CREATE TABLE invocations ("
            "job_key TEXT REFERENCES tool_keys(job_key), ordinal INTEGER, phase TEXT NOT NULL, "
            "PRIMARY KEY(job_key, ordinal));"
            "CREATE TABLE effects (job_key TEXT PRIMARY KEY REFERENCES tool_keys(job_key), "
            "payload TEXT NOT NULL, applications INTEGER NOT NULL);"
        )
        db.executemany(
            "INSERT INTO tool_keys VALUES (?)", [(key,) for key, _ in _WORKLOAD]
        )
        db.commit()


def _record_invocation(directory: Path, key: str, kind: str, phase: str) -> int:
    # This independent durable ledger is a mock external service. Idempotent
    # effects deduplicate by key; non-idempotent effects count every application.
    with closing(sqlite3.connect(directory / "effects.sqlite3", timeout=5)) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("BEGIN IMMEDIATE")
        ordinal = db.execute(
            "SELECT COUNT(*) + 1 FROM invocations WHERE job_key=?", (key,)
        ).fetchone()[0]
        db.execute("INSERT INTO invocations VALUES (?, ?, ?)", (key, ordinal, phase))
        if kind != "fatal" and not (kind == "flaky" and ordinal == 1):
            payload = json.dumps({"key": key, "kind": kind}, sort_keys=True)
            db.execute(
                "INSERT INTO effects VALUES (?, ?, 1) ON CONFLICT(job_key) DO UPDATE "
                "SET applications=applications + ?",
                (key, payload, int(not _POLICIES[kind].idempotent)),
            )
        db.commit()
        return ordinal


def _registry(
    directory: Path, phase: str, entered: dict[str, asyncio.Event]
) -> ToolRegistry:
    registry = ToolRegistry()

    def make_tool(kind: str):
        async def invoke(key: str) -> JSONValue:
            ordinal = await _offload(_record_invocation, directory, key, kind, phase)
            if kind.endswith("blocker") and phase == "crash":
                entered[key].set()
                await asyncio.Event().wait()
            if kind == "flaky" and ordinal == 1:
                raise ToolRetryableError("ledger-confirmed first invocation")
            if kind == "fatal":
                raise ToolFatalError("definitive rejection; no side effect")
            if kind == "slow":
                await asyncio.sleep(0.01)
            return {"key": key, "kind": kind}

        return invoke

    for kind, policy in _POLICIES.items():
        registry.register(policy, make_tool(kind))
    return registry


def _accepted_ids(message: dict[str, Any]) -> list[str]:
    ids = message.get("accepted_ids")
    _require(
        isinstance(ids, list)
        and len(ids) == len(_WORKLOAD)
        and all(isinstance(job_id, str) and job_id for job_id in ids)
        and len(set(ids)) == len(ids),
        "child must acknowledge exactly 100 distinct committed job IDs",
    )
    return cast(list[str], ids)


async def _child_main(directory: Path, phase: str, mode: str, timeout: float) -> None:
    db_path = directory / "jobs.sqlite3"
    _require(
        db_path.exists() == (phase == "recover"),
        "unexpected internal child database path",
    )
    entered = {
        key: asyncio.Event() for key, kind in _WORKLOAD if kind.endswith("blocker")
    }
    config = RuntimeConfig(
        db_path=str(db_path),
        max_workers=8,
        max_outstanding_jobs=100,
        lease_duration_ms=2000,
        heartbeat_interval_ms=200,
        poll_interval_ms=10,
        recovery_interval_ms=20,
        synchronous_mode=cast(Literal["FULL", "NORMAL"], mode),
    )
    async with asyncio.timeout(timeout):
        async with DurableToolRuntime(
            config, _registry(directory, phase, entered)
        ) as runtime:
            if phase == "crash":
                ids = []
                for index, (key, kind) in enumerate(_WORKLOAD):
                    if index == 40:
                        completed = await asyncio.gather(
                            *(runtime.wait_for_terminal(job_id) for job_id in ids)
                        )
                        _require(
                            all(job.status is JobStatus.SUCCEEDED for job in completed),
                            "fast jobs failed",
                        )
                    if index == 48:
                        # All eight workers have committed their ledger effects
                        # and are now occupied indefinitely, before admitting the tail.
                        await asyncio.gather(
                            *(event.wait() for event in entered.values())
                        )
                    ids.append(await runtime.submit(kind, {"key": key}, key))
                print(json.dumps({"phase": phase, "accepted_ids": ids}), flush=True)
                await asyncio.Event().wait()
            else:
                message = json.loads(
                    await _offload((directory / "manifest.json").read_text)
                )
                ids = _accepted_ids(message)
                await asyncio.gather(
                    *(runtime.wait_for_terminal(job_id) for job_id in ids)
                )
        print(json.dumps({"phase": phase, "accepted_ids": ids}), flush=True)


@dataclass
class _Snapshot:
    jobs: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]
    invocations: list[dict[str, Any]]
    effects: dict[str, dict[str, Any]]
    integrity: bool
    foreign_keys: bool


def _snapshot(directory: Path, check: bool = False) -> _Snapshot:
    # Read-only connections neither run the runtime's recovery nor request WAL
    # checkpoints. In particular, WAL presence is stat'ed BEFORE this post-crash.
    data: dict[str, list[dict[str, Any]]] = {}
    integrity = foreign_keys = True
    for filename, tables in (
        ("jobs.sqlite3", ("jobs", "job_events")),
        ("effects.sqlite3", ("invocations", "effects")),
    ):
        uri = (directory / filename).as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=1)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            for table in tables:
                data[table] = [
                    dict(row) for row in db.execute(f"SELECT * FROM {table}")
                ]
            if check:
                integrity &= [
                    tuple(row) for row in db.execute("PRAGMA integrity_check")
                ] == [("ok",)]
                foreign_keys &= not db.execute("PRAGMA foreign_key_check").fetchall()
    return _Snapshot(
        {row["job_id"]: row for row in data["jobs"]},
        data["job_events"],
        data["invocations"],
        {row["job_key"]: row for row in data["effects"]},
        integrity,
        foreign_keys,
    )


@dataclass
class _Child:
    process: asyncio.subprocess.Process
    line: asyncio.Task[bytes]
    stderr: asyncio.Task[bytes]


async def _finish_cleanup(task: asyncio.Task):
    # A second cancellation must not abandon a spawning or reaping subprocess.
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _offload(function, *args):
    # Finish file/SQLite access before cancellation can remove the temp directory.
    return await _finish_cleanup(
        asyncio.create_task(asyncio.to_thread(function, *args))
    )


async def _spawn_child(
    children: list[_Child], directory: Path, phase: str, mode: str, timeout: float
) -> _Child:
    async def spawn() -> _Child:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "benchmarks.chaos_crash_test",
            "--internal-child",
            phase,
            "--internal-directory",
            str(directory),
            "--synchronous",
            mode,
            "--internal-timeout",
            str(timeout),
            cwd=Path(__file__).resolve().parents[1],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        child = _Child(
            process,
            asyncio.create_task(process.stdout.readline()),
            asyncio.create_task(process.stderr.read()),
        )
        # Register ownership inside the shielded spawn, closing the cancellation
        # window between OS process creation and the caller receiving its handle.
        children.append(child)
        return child

    return await _finish_cleanup(asyncio.create_task(spawn()))


async def _read_message(child: _Child, phase: str) -> dict[str, Any]:
    line = await child.line
    if not line:
        error = (await child.stderr).decode(errors="replace")
        raise RuntimeError(f"{phase} child exited before its acknowledgement: {error}")
    message = json.loads(line)
    _require(
        isinstance(message, dict) and message.get("phase") == phase,
        "invalid child protocol message",
    )
    return message


async def _cleanup_children(children: list[_Child]) -> None:
    for child in children:
        if child.process.returncode is None:
            with suppress(ProcessLookupError):
                child.process.kill()  # Only handles created and owned by this invocation.
    if children:
        await asyncio.wait_for(
            asyncio.gather(*(child.process.wait() for child in children)),
            timeout=5,
        )
        await asyncio.wait_for(
            asyncio.gather(
                *(task for child in children for task in (child.line, child.stderr)),
                return_exceptions=True,
            ),
            timeout=5,
        )


async def _wait_crash_point(
    directory: Path, ids: list[str], child: _Child
) -> _Snapshot:
    expected = {
        "PENDING": 52,
        "LEASED": 8,
        "RETRY_WAIT": 0,
        "SUCCEEDED": 40,
        "FAILED": 0,
        "UNKNOWN_OUTCOME": 0,
    }
    while True:
        if child.process.returncode is not None:
            error = (await child.stderr).decode(errors="replace")
            raise RuntimeError(f"crash child exited before SIGKILL: {error}")
        snapshot = await _offload(_snapshot, directory)
        _require(
            set(snapshot.jobs) == set(ids), "acknowledged IDs differ from committed IDs"
        )
        if _counts(snapshot.jobs) == expected:
            leased = [
                row for row in snapshot.jobs.values() if row["status"] == "LEASED"
            ]
            if Counter(row["tool_name"] for row in leased) == {
                "idem_blocker": 4,
                "nonidem_blocker": 4,
            }:
                calls = Counter(row["job_key"] for row in snapshot.invocations)
                if all(
                    calls[key] == 1 and key in snapshot.effects
                    for key, _ in _WORKLOAD[:48]
                ):
                    return snapshot
        await asyncio.sleep(0.01)


def _verify(
    mode: str,
    ids: list[str],
    before: _Snapshot,
    after: _Snapshot,
    killed: int,
    recovered: int,
    wal: bool,
    elapsed: float,
) -> ChaosReport:
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        _require(condition, f"chaos verification failed: {name}")
        checks[name] = True

    check("child_exit_codes", killed == -signal.SIGKILL and recovered == 0)
    check("post_crash_wal_observed", wal)
    check(
        "accepted_ids_exactly_once",
        len(ids) == len(set(ids)) == 100
        and set(ids) == set(before.jobs) == set(after.jobs),
    )
    check("sqlite_integrity", after.integrity)
    check("sqlite_foreign_keys", after.foreign_keys)
    check(
        "expected_terminal_states",
        _counts(after.jobs)
        == {
            "PENDING": 0,
            "LEASED": 0,
            "RETRY_WAIT": 0,
            "SUCCEEDED": 88,
            "FAILED": 8,
            "UNKNOWN_OUTCOME": 4,
        },
    )
    before_calls = Counter(row["job_key"] for row in before.invocations)
    after_calls = Counter(row["job_key"] for row in after.invocations)
    check(
        "pending_not_invoked_before_crash",
        set(before_calls) == {key for key, _ in _WORKLOAD[:48]},
    )
    check("ledger_keys", set(after_calls) == {key for key, _ in _WORKLOAD})
    precompleted = reexecutions = unknown_reexecutions = 0
    events: dict[str, list[dict[str, Any]]] = {job_id: [] for job_id in ids}
    for event in sorted(after.events, key=lambda row: row["event_id"]):
        check("audit_job_ids", event["job_id"] in events)
        events[event["job_id"]].append(event)
    for index, ((key, kind), job_id) in enumerate(zip(_WORKLOAD, ids, strict=True)):
        old, new = before.jobs[job_id], after.jobs[job_id]
        policy = _POLICIES[kind]
        check(
            "manifest_matches_jobs",
            new["idempotency_key"] == key
            and new["tool_name"] == kind
            and json.loads(new["args_json"]) == {"key": key},
        )
        fields = ("idempotent", "timeout_ms", "max_retries", "base_backoff_ms")
        check(
            "policies_unchanged",
            all(old[field] == new[field] == getattr(policy, field) for field in fields),
        )
        attempts = 2 if kind in ("idem_blocker", "flaky") else 1
        check(
            "ledger_matches_attempts",
            new["attempt_count"] == after_calls[key] == attempts,
        )
        expected_status = (
            "FAILED"
            if kind == "fatal"
            else "UNKNOWN_OUTCOME"
            if kind == "nonidem_blocker"
            else "SUCCEEDED"
        )
        check("per_job_terminal_state", new["status"] == expected_status)
        path = ["PENDING", "LEASED"]
        if attempts == 2:
            path += ["RETRY_WAIT", "LEASED"]
        path += [expected_status]
        audit = events[job_id]
        check(
            "audit_paths",
            [event["to_status"] for event in audit] == path
            and all(event["event_type"] == "STATE_TRANSITION" for event in audit),
        )
        check(
            "audit_attempts",
            [event["attempt"] for event in audit]
            == ([0, 1, 1, 2, 2] if attempts == 2 else [0, 1, 1]),
        )
        phases = Counter(
            row["phase"] for row in after.invocations if row["job_key"] == key
        )
        check(
            "invocation_phases",
            phases
            == Counter(
                {"crash": 1, "recover": attempts - 1}
                if index < 48
                else {"recover": attempts}
            ),
        )
        if index < 40:
            check(
                "precompleted_unchanged",
                old == new
                and old["status"] == "SUCCEEDED"
                and old["attempt_count"] == 1,
            )
            precompleted += 1
            reexecutions += after_calls[key] - before_calls[key]
        if kind == "nonidem_blocker":
            unknown_reexecutions += after_calls[key] - before_calls[key]
            check(
                "unknown_effect_before_crash",
                before.effects.get(key) == after.effects.get(key)
                and key in before.effects,
            )
        if expected_status == "SUCCEEDED":
            check(
                "results", json.loads(new["result_json"]) == {"key": key, "kind": kind}
            )
        if kind == "fatal":
            check(
                "fatal_no_effect",
                key not in after.effects and "ToolFatalError" in new["error_text"],
            )
        else:
            effect = after.effects[key]
            check(
                "deduplicated_effects",
                effect["applications"] == 1
                and json.loads(effect["payload"]) == {"key": key, "kind": kind},
            )
    check("precompleted_not_reexecuted", reexecutions == 0)
    check("unknown_not_replayed", unknown_reexecutions == 0)
    check(
        "effect_keys",
        set(after.effects) == {key for key, kind in _WORKLOAD if kind != "fatal"},
    )
    evidence = {
        "precompleted_preserved": precompleted,
        "precompleted_reexecutions": reexecutions,
        "unknown_reexecutions": unknown_reexecutions,
        "nonidem_effects_before_crash": sum(
            key in before.effects
            for key, kind in _WORKLOAD
            if kind == "nonidem_blocker"
        ),
        "idem_blocker_calls": sum(
            after_calls[key] for key, kind in _WORKLOAD if kind == "idem_blocker"
        ),
        "idem_blocker_effects": sum(
            key in after.effects for key, kind in _WORKLOAD if kind == "idem_blocker"
        ),
        "flaky_calls": sum(
            after_calls[key] for key, kind in _WORKLOAD if kind == "flaky"
        ),
        "flaky_effects": sum(
            key in after.effects for key, kind in _WORKLOAD if kind == "flaky"
        ),
    }
    return ChaosReport(
        mode,
        len(ids),
        _counts(before.jobs),
        _counts(after.jobs),
        killed,
        recovered,
        wal,
        evidence,
        checks,
        elapsed,
    )


async def run_chaos(synchronous_mode: str = "FULL", timeout: float = 30) -> ChaosReport:
    """Kill one owned child, recover in a fresh child, and verify both databases.

    ``timeout`` is a cooperative deadline, not a hard wall-clock bound: expiry
    waits for cancellation-safe spawn and offloaded I/O joins, then bounded child
    reaping, all excluded from the deadline. No database path is accepted: every
    run owns a fresh tempdir. Requires POSIX SIGKILL; raises on a timeout or any
    failed verification.
    """
    if synchronous_mode not in ("FULL", "NORMAL"):
        raise ValueError("synchronous_mode must be FULL or NORMAL")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    if os.name != "posix" or not hasattr(signal, "SIGKILL"):
        raise RuntimeError("the chaos harness requires POSIX SIGKILL")
    started = time.perf_counter()
    children: list[_Child] = []
    with tempfile.TemporaryDirectory(prefix="durable-chaos-") as temporary:
        directory = Path(temporary)
        try:
            async with asyncio.timeout(timeout):
                await _offload(_init_ledger, directory)
                crash = await _spawn_child(
                    children, directory, "crash", synchronous_mode, timeout
                )
                ids = _accepted_ids(await _read_message(crash, "crash"))
                before = await _wait_crash_point(directory, ids, crash)
                crash.process.kill()
                killed = (
                    await crash.process.wait()
                )  # Reap before starting a second process.
                _require(
                    killed == -signal.SIGKILL, "child was not terminated by SIGKILL"
                )
                # Do not open either SQLite DB before observing the crashed WAL.
                wal_path = directory / "jobs.sqlite3-wal"
                wal = wal_path.is_file() and wal_path.stat().st_size > 0
                await _offload(
                    (directory / "manifest.json").write_text,
                    json.dumps({"accepted_ids": ids}),
                )
                recovery = await _spawn_child(
                    children, directory, "recover", synchronous_mode, timeout
                )
                recovered_ids = _accepted_ids(await _read_message(recovery, "recover"))
                recovered = await recovery.process.wait()
                _require(
                    recovered_ids == ids,
                    "recovery child did not conserve the acknowledged manifest",
                )
                _require(
                    recovered == 0,
                    f"recovery child failed: {(await recovery.stderr).decode(errors='replace')}",
                )
                after = await _offload(_snapshot, directory, True)
                return _verify(
                    synchronous_mode,
                    ids,
                    before,
                    after,
                    killed,
                    recovered,
                    wal,
                    time.perf_counter() - started,
                )
        finally:
            await _finish_cleanup(asyncio.create_task(_cleanup_children(children)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synchronous", choices=("FULL", "NORMAL"), default="FULL")
    parser.add_argument(
        "--json", type=Path, help="write the structured verification report"
    )
    parser.add_argument(
        "--internal-child", choices=("crash", "recover"), help=argparse.SUPPRESS
    )
    parser.add_argument("--internal-directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-timeout", type=float, default=30, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.internal_child:
        if args.internal_directory is None or not args.internal_directory.is_dir():
            parser.error("internal child requires an existing --internal-directory")
        asyncio.run(
            _child_main(
                args.internal_directory,
                args.internal_child,
                args.synchronous,
                args.internal_timeout,
            )
        )
        return
    if args.internal_directory is not None:
        parser.error("--internal-directory is only for internal child processes")
    report = asyncio.run(run_chaos(args.synchronous))
    if args.json is not None:
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(format_report(report))


if __name__ == "__main__":
    main()
