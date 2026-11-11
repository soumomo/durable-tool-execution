from __future__ import annotations

import asyncio
import copy
import json
import os
import signal
import threading
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks import chaos_crash_test as chaos

_POSIX = os.name == "posix" and hasattr(signal, "SIGKILL")


@pytest.fixture
def owned_children(monkeypatch):
    children = []
    directories = set()
    spawn = chaos._spawn_child

    async def track(*args):
        child = await spawn(*args)
        directories.add(args[1])
        children.append(child)
        return child

    monkeypatch.setattr(chaos, "_spawn_child", track)
    yield children, directories
    assert all(child.process.returncode is not None for child in children)
    assert all(child.line.done() and child.stderr.done() for child in children)
    assert all(not directory.exists() for directory in directories)


@pytest.fixture
def expire_deadline(monkeypatch):
    deadline = asyncio.Timeout(None)
    monkeypatch.setattr(chaos.asyncio, "timeout", lambda _: deadline)

    async def expire():
        observed = asyncio.Event()
        loop = asyncio.get_running_loop()
        deadline.reschedule(loop.time())
        # Observe expiry after the run task has had a turn to handle cancellation.
        loop.call_soon(observed.set)
        await observed.wait()
        assert deadline.expired()

    return expire


@pytest.mark.parametrize("mode", ["FULL", "NORMAL"])
@pytest.mark.chaos
@pytest.mark.skipif(not _POSIX, reason="requires POSIX SIGKILL")
async def test_real_sigkill_recovery_and_report(
    mode, owned_children, monkeypatch, tmp_path, capsys
):
    verification = []
    verify = chaos._verify

    def capture(*args):
        verification.append(args)
        return verify(*args)

    monkeypatch.setattr(chaos, "_verify", capture)
    report = await chaos.run_chaos(synchronous_mode=mode)
    children, _ = owned_children
    assert len(children) == 2
    assert children[0].process.pid != children[1].process.pid
    assert [child.process.returncode for child in children] == [-signal.SIGKILL, 0]
    assert report.synchronous_mode == mode
    assert report.before_crash == {
        "PENDING": 52,
        "LEASED": 8,
        "RETRY_WAIT": 0,
        "SUCCEEDED": 40,
        "FAILED": 0,
        "UNKNOWN_OUTCOME": 0,
    }
    assert report.after_recovery == {
        "PENDING": 0,
        "LEASED": 0,
        "RETRY_WAIT": 0,
        "SUCCEEDED": 88,
        "FAILED": 8,
        "UNKNOWN_OUTCOME": 4,
    }
    assert report.accepted == 100
    assert report.conserved and report.unfinished == 0
    assert report.wal_present_after_sigkill
    assert report.elapsed_seconds > 0
    assert report.checks and all(report.checks.values())
    assert report.evidence == {
        "precompleted_preserved": 40,
        "precompleted_reexecutions": 0,
        "unknown_reexecutions": 0,
        "nonidem_effects_before_crash": 4,
        "idem_blocker_calls": 8,
        "idem_blocker_effects": 4,
        "flaky_calls": 40,
        "flaky_effects": 20,
    }

    # Exercise the real CLI formatter/JSON writer with this measured report,
    # without running another process-kill experiment just for serialization.
    async def measured_report(synchronous_mode="FULL", timeout=30):
        assert synchronous_mode == mode
        return report

    monkeypatch.setattr(chaos, "run_chaos", measured_report)
    output_path = tmp_path / "report.json"
    await asyncio.to_thread(
        chaos.main, ["--synchronous", mode, "--json", str(output_path)]
    )
    output = capsys.readouterr().out
    assert output == chaos.format_report(report) + "\n"
    assert "100 accepted = 88 succeeded + 8 failed + 4 unknown (PASS)" in output
    assert "integrity_check=ok; foreign_key_check=clean" in output
    assert "40 unchanged; 0 reexecutions" in output
    assert str(tmp_path) not in output
    assert "pid=" not in output.lower()
    structured = json.loads(output_path.read_text())
    assert structured == report.to_dict()
    assert structured["conserved"] is True and structured["unfinished"] == 0

    assert not replace(report, accepted=101).conserved
    unfinished = replace(report, after_recovery={**report.after_recovery, "PENDING": 1})
    assert unfinished.unfinished == 1 and not unfinished.conserved
    unchecked = replace(
        report, checks={**report.checks, "accepted_ids_exactly_once": False}
    )
    assert not unchecked.conserved
    assert "(FAIL)" in chaos.format_report(unchecked)

    # Corrupt only in-memory copies of observations from the real run. Never
    # fabricate persistent states or mutate the runtime DB to simulate a crash.
    observed_mode, ids, before, after, killed, recovered, wal, elapsed = verification[0]

    def verify_after(observation, accepted_ids=ids):
        return verify(
            observed_mode,
            accepted_ids,
            before,
            observation,
            killed,
            recovered,
            wal,
            elapsed,
        )

    assert set(before.jobs) == set(after.jobs) == set(ids)
    for field, expected_check in (
        ("result_json", "precompleted_unchanged"),
        ("attempt_count", "ledger_matches_attempts"),
        ("max_retries", "policies_unchanged"),
    ):
        altered = copy.deepcopy(after)
        job = altered.jobs[ids[0]]
        job[field] = '"changed"' if field == "result_json" else job[field] + 1
        with pytest.raises(AssertionError, match=expected_check):
            verify_after(altered)
    for index in (0, 40):  # Completed replay and UNKNOWN_OUTCOME replay.
        altered = copy.deepcopy(after)
        key = chaos._WORKLOAD[index][0]
        altered.invocations.append({"job_key": key, "ordinal": 2, "phase": "recover"})
        with pytest.raises(AssertionError, match="ledger_matches_attempts"):
            verify_after(altered)
    for field, expected_check in (
        ("integrity", "sqlite_integrity"),
        ("foreign_keys", "sqlite_foreign_keys"),
    ):
        altered = replace(after, **{field: False})
        with pytest.raises(AssertionError, match=expected_check):
            verify_after(altered)
    with pytest.raises(AssertionError, match="accepted_ids_exactly_once"):
        verify_after(after, ids[:-1] + ids[:1])


@pytest.mark.parametrize("phase", ["crash", "recover"])
@pytest.mark.chaos
@pytest.mark.skipif(not _POSIX, reason="requires POSIX SIGKILL")
async def test_cancellation_reaps_every_owned_child_even_when_cancelled_twice(
    phase, owned_children, monkeypatch
):
    started, cleanup_started, release_cleanup = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    spawn, cleanup = chaos._spawn_child, chaos._cleanup_children

    async def pause_after_spawn(*args):
        child = await spawn(*args)
        if args[2] == phase:
            started.set()
            await asyncio.Event().wait()
        return child

    async def pause_cleanup(children):
        cleanup_started.set()
        await release_cleanup.wait()
        await cleanup(children)

    monkeypatch.setattr(chaos, "_spawn_child", pause_after_spawn)
    monkeypatch.setattr(chaos, "_cleanup_children", pause_cleanup)
    task = asyncio.create_task(chaos.run_chaos())
    try:
        await asyncio.wait_for(started.wait(), 15)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 5)
        task.cancel()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
    finally:
        release_cleanup.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    children, _ = owned_children
    assert len(children) == (1 if phase == "crash" else 2)
    assert all(child.process.returncode == -signal.SIGKILL for child in children)


@pytest.mark.chaos
@pytest.mark.skipif(not _POSIX, reason="requires POSIX SIGKILL")
async def test_exception_after_recovery_spawn_reaps_both_children(
    owned_children, monkeypatch
):
    spawn = chaos._spawn_child

    async def fail_after_spawn(*args):
        child = await spawn(*args)
        if args[2] == "recover":
            raise RuntimeError("injected parent observation failure")
        return child

    monkeypatch.setattr(chaos, "_spawn_child", fail_after_spawn)
    with pytest.raises(RuntimeError, match="injected parent observation failure"):
        await chaos.run_chaos()
    children, _ = owned_children
    assert len(children) == 2
    assert all(child.process.returncode == -signal.SIGKILL for child in children)


@pytest.mark.chaos
@pytest.mark.skipif(not _POSIX, reason="requires POSIX SIGKILL")
async def test_whole_run_deadline_reaps_child(owned_children, monkeypatch):
    spawn = chaos._spawn_child

    async def pause_after_spawn(*args):
        await spawn(*args)
        await asyncio.Event().wait()

    monkeypatch.setattr(chaos, "_spawn_child", pause_after_spawn)
    with pytest.raises(TimeoutError):
        await chaos.run_chaos(timeout=2)
    children, _ = owned_children
    assert len(children) == 1 and children[0].process.returncode == -signal.SIGKILL


@pytest.mark.chaos
@pytest.mark.skipif(not _POSIX, reason="requires POSIX SIGKILL")
async def test_deadline_during_offloaded_io_joins_before_removing_directory(
    monkeypatch, expire_deadline, owned_children
):
    started = asyncio.Event()
    release, finished = threading.Event(), threading.Event()
    directories = []
    loop = asyncio.get_running_loop()
    init = chaos._init_ledger

    def delayed_init(directory):
        directories.append(directory)
        loop.call_soon_threadsafe(started.set)
        release.wait()
        assert directory.is_dir()
        init(directory)
        finished.set()

    monkeypatch.setattr(chaos, "_init_ledger", delayed_init)
    task = asyncio.create_task(chaos.run_chaos())
    try:
        await asyncio.wait_for(started.wait(), 10)
        await expire_deadline()
        assert task.cancelling() and not task.done()
        assert directories[0].is_dir() and not finished.is_set()
        release.set()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, 10)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, TimeoutError):
            await task
    assert finished.is_set()
    assert not directories[0].exists()
    children, _ = owned_children
    assert not children


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
@pytest.mark.chaos
@pytest.mark.skipif(not _POSIX, reason="requires POSIX SIGKILL")
async def test_interruption_during_os_spawn_still_registers_and_reaps_child(
    monkeypatch, expire_deadline, interruption
):
    created, release = asyncio.Event(), asyncio.Event()
    processes, directories = [], []
    spawn = asyncio.create_subprocess_exec

    async def delayed_spawn(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        directories.append(Path(args[args.index("--internal-directory") + 1]))
        created.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    task = asyncio.create_task(chaos.run_chaos())
    try:
        await asyncio.wait_for(created.wait(), 10)
        if interruption == "deadline":
            await expire_deadline()
            assert task.cancelling() and not task.done()
            assert processes[0].returncode is None and directories[0].is_dir()
        else:
            task.cancel()
        release.set()
        with pytest.raises(
            TimeoutError if interruption == "deadline" else asyncio.CancelledError
        ):
            await asyncio.wait_for(task, 10)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, TimeoutError):
            await task
    assert len(processes) == 1 and processes[0].returncode == -signal.SIGKILL
    assert all(not directory.exists() for directory in directories)


async def test_unsupported_platform_is_rejected_before_spawn(monkeypatch):
    monkeypatch.delattr(chaos.signal, "SIGKILL", raising=False)
    with pytest.raises(RuntimeError, match="POSIX SIGKILL"):
        await chaos.run_chaos()


@pytest.mark.parametrize("mode", ["full", "OFF", "", None])
async def test_invalid_synchronous_mode_is_rejected_before_spawn(mode):
    with pytest.raises(ValueError, match="synchronous_mode"):
        await chaos.run_chaos(synchronous_mode=mode)


@pytest.mark.parametrize(
    "timeout", [0, -1, float("inf"), float("nan"), True, "30", None]
)
async def test_invalid_deadline_is_rejected_before_spawn(timeout):
    with pytest.raises(ValueError, match="timeout"):
        await chaos.run_chaos(timeout=timeout)


@pytest.mark.parametrize(
    "ids", [None, [], ["same"] * 100, list(range(100)), [""] * 100]
)
def test_manifest_requires_100_distinct_nonempty_ids(ids):
    with pytest.raises(AssertionError, match="100 distinct committed job IDs"):
        chaos._accepted_ids({"accepted_ids": ids})


def test_internal_child_flags_are_hidden_and_require_a_directory(capsys):
    with pytest.raises(SystemExit) as help_exit:
        chaos.main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--internal" not in help_text
    assert "--synchronous" in help_text and "--json" in help_text
    with pytest.raises(SystemExit) as missing_directory:
        chaos.main(["--internal-child", "crash"])
    assert missing_directory.value.code == 2


def test_cli_does_not_accept_a_user_database_path():
    with pytest.raises(SystemExit) as error:
        chaos.main(["--db-path", str(Path("do-not-touch.sqlite3"))])
    assert error.value.code == 2
