"""Contract tests for the lightweight bridge and key-free, MOCK-only CLI."""

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import pytest_asyncio

from examples.react_agent_bridge import ReActToolBridge, main, run_demo, tool_call_key
from runtime import (
    DurableToolRuntime,
    IdempotencyConflictError,
    JobStatus,
    RuntimeConfig,
    ToolFatalError,
    ToolRegistry,
)


@pytest_asyncio.fixture
async def runtime_factory(tmp_path):
    runtimes = []

    async def make(registry, **overrides):
        options = {
            "db_path": str(tmp_path / f"bridge-{len(runtimes)}.sqlite3"),
            "max_workers": 3,
            "poll_interval_ms": 5,
            "recovery_interval_ms": 10,
        }
        options.update(overrides)
        runtime = DurableToolRuntime(RuntimeConfig(**options), registry)
        runtimes.append(runtime)
        await runtime.start()
        return runtime

    yield make
    for runtime in reversed(runtimes):
        await asyncio.wait_for(runtime.stop(graceful=False), timeout=5)


@pytest.fixture
def tools():
    registry = ToolRegistry()
    effects = []

    @registry.tool(idempotent=True, max_retries=0)
    async def simulate_move(move):
        effects.append(("simulate_move", move))
        return {"mock": True, "move": move}

    @registry.tool(idempotent=False, max_retries=0)
    async def play_move(move):
        effects.append(("play_move", move))
        return {"mock": True, "move": move}

    @registry.tool(idempotent=False, max_retries=0)
    async def execute_bash(command):
        effects.append(("execute_bash", command))
        return {"mock": True, "command": command}

    return registry, effects


def call(call_id="call-1", name="play_move", **args):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def payload(observation):
    assert observation["role"] == "tool"
    assert isinstance(observation["content"], str)
    return json.loads(observation["content"])


async def execute(bridge, calls, **options):
    options = {"run_id": "test-run", "step_idx": 0, **options}
    return await asyncio.wait_for(
        bridge.execute_tool_calls(calls, **options), timeout=5
    )


def test_stable_key_is_sha256_of_structured_json_and_not_arguments():
    envelope = ["run:雪", 2, "call:α"]
    expected = hashlib.sha256(
        json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert tool_call_key(*envelope) == expected
    assert len(expected) == 64
    assert tool_call_key(*envelope) == tool_call_key(*envelope)
    assert tool_call_key("another-run", 2, "call:α") != expected
    assert tool_call_key("run:雪", 3, "call:α") != expected
    assert tool_call_key("run:雪", 2, "other-call") != expected


def test_delimiter_collision_cannot_alias_keys():
    left = ("run:1", 2, "call")
    right = ("run", 1, "2:call")
    assert ":".join(map(str, left)) == ":".join(map(str, right))
    assert tool_call_key(*left) != tool_call_key(*right)
    assert tool_call_key("run", 0, "call") != tool_call_key("run ", 0, "call")


@pytest.mark.parametrize(
    "run_id,step_idx,call_id",
    [
        (None, 0, "call"),
        (7, 0, "call"),
        ("", 0, "call"),
        ("  ", 0, "call"),
        ("run", -1, "call"),
        ("run", True, "call"),
        ("run", 1.0, "call"),
        ("run", "1", "call"),
        ("run", None, "call"),
        ("run", 0, ""),
        ("run", 0, " \t"),
        ("run", 0, None),
        ("run", 0, 12),
    ],
)
def test_invalid_key_envelopes(run_id, step_idx, call_id):
    with pytest.raises((ValueError, TypeError)):
        tool_call_key(run_id, step_idx, call_id)


async def test_linking_order_and_duplicate_execution_uses_cache(runtime_factory, tools):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    calls = [
        call("move", move="e2e4"),
        call("shell", "execute_bash", command="MOCK text"),
    ]
    first = await execute(bridge, calls)
    replay = await execute(bridge, calls)
    assert replay == first
    assert [o["tool_call_id"] for o in first] == ["move", "shell"]
    assert effects == [("play_move", "e2e4"), ("execute_bash", "MOCK text")]
    jobs = await runtime.store.list_jobs()
    assert len(jobs) == 2
    for observation, job in zip(first, jobs, strict=True):
        assert payload(observation) == {
            "job_id": job.job_id,
            "status": "SUCCEEDED",
            "result": job.result,
        }
        assert job.attempt_count == 1
        assert job.idempotency_key == tool_call_key(
            "test-run", 0, observation["tool_call_id"]
        )


async def test_cached_replay_survives_new_runtime_and_bridge(runtime_factory, tools):
    registry, effects = tools
    original = await runtime_factory(registry)
    calls = [call("move", move="e2e4")]
    first = await execute(ReActToolBridge(original), calls)
    await original.stop()
    restarted = await runtime_factory(registry, db_path=original.config.db_path)
    assert await execute(ReActToolBridge(restarted), calls) == first
    assert effects == [("play_move", "e2e4")]


async def test_same_call_id_in_different_run_or_step_is_distinct(
    runtime_factory, tools
):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    calls = [call(move="e2e4")]
    replies = [
        await execute(bridge, calls),
        await execute(bridge, calls, step_idx=1),
        await execute(bridge, calls, run_id="other-run"),
    ]
    assert len({payload(reply[0])["job_id"] for reply in replies}) == 3
    assert effects == [("play_move", "e2e4")] * 3


async def test_canonical_argument_order_reuses_job(runtime_factory):
    registry = ToolRegistry()
    effects = []

    @registry.tool(idempotent=False)
    async def mutate(a, b):
        effects.append((a, b))

    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    original = call("mutate", "mutate", a=1, b={"x": 2, "y": 3})
    reordered = call("mutate", "mutate", b={"y": 3, "x": 2}, a=1)
    first = await execute(bridge, [original])
    assert await execute(bridge, [reordered]) == first
    assert payload(first[0])["result"] is None
    assert len(effects) == 1


@pytest.mark.parametrize(
    "changed",
    [
        call("old", move="d2d4"),
        call("old", "simulate_move", move="e2e4"),
    ],
)
async def test_existing_key_conflict_is_preflighted_before_new_mutation(
    runtime_factory,
    tools,
    changed,
):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    await execute(bridge, [call("old", move="e2e4")])
    # The new mutation is deliberately before the conflicting historical key.
    with pytest.raises(IdempotencyConflictError):
        await execute(bridge, [call("new", "execute_bash", command="MOCK"), changed])
    assert effects == [("play_move", "e2e4")]
    assert len(await runtime.store.list_jobs()) == 1


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "not-a-dict",
        {},
        {"id": ""},
        {"id": 17},
        {"id": "bad", "function": None},
        {
            "id": "bad",
            "type": "other",
            "function": {"name": "play_move", "arguments": "{}"},
        },
        {"id": "bad", "function": {"name": "", "arguments": "{}"}},
        {"id": "bad", "function": {"name": "missing", "arguments": "{}"}},
        {"id": "bad", "function": {"name": "play_move"}},
        {"id": "bad", "function": {"name": "play_move", "arguments": {"move": "e2e4"}}},
        {"id": "bad", "function": {"name": "play_move", "arguments": "not-json"}},
        {"id": "bad", "function": {"name": "play_move", "arguments": "[]"}},
        {"id": "bad", "function": {"name": "play_move", "arguments": "null"}},
        {"id": "bad", "function": {"name": "play_move", "arguments": "3"}},
        call("bad", "play_move"),
        call("bad", "play_move", move="e2e4", extra=True),
        {"id": "bad", "function": {"name": "play_move", "arguments": '{"move": NaN}'}},
        {
            "id": "bad",
            "function": {"name": "play_move", "arguments": '{"move": Infinity}'},
        },
        {
            "id": "bad",
            "function": {"name": "play_move", "arguments": '{"move": 1e999}'},
        },
        {
            "id": "bad",
            "function": {
                "name": "play_move",
                "arguments": '{"move": "a", "move": "b"}',
            },
        },
        {
            "id": "bad",
            "function": {
                "name": "play_move",
                "arguments": '{"move": {"x": 1, "x": 2}}',
            },
        },
        {
            "id": "bad",
            "function": {"name": "play_move", "arguments": '{"move": "\\ud800"}'},
        },
    ],
)
async def test_invalid_later_call_causes_no_partial_effects(
    runtime_factory, tools, bad
):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    with pytest.raises((ValueError, TypeError)):
        await execute(
            bridge, [call("valid-first", "execute_bash", command="MOCK"), bad]
        )
    assert effects == []
    assert await runtime.store.list_jobs() == []


async def test_duplicate_ids_are_rejected_before_any_admission(runtime_factory, tools):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    with pytest.raises(ValueError, match="Duplicate call ID"):
        await execute(
            bridge,
            [
                call("duplicate", "execute_bash", command="MOCK"),
                call("duplicate", "simulate_move", move="e2e4"),
            ],
        )
    assert effects == []
    assert await runtime.store.list_jobs() == []


async def test_multiple_moves_in_one_turn_are_rejected_upfront(runtime_factory, tools):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    with pytest.raises(ValueError, match="Only one play_move"):
        await execute(
            ReActToolBridge(runtime),
            [call("one", move="e2e4"), call("two", move="e7e5")],
        )
    assert effects == []
    assert await runtime.store.list_jobs() == []


@pytest.mark.parametrize(
    "options",
    [
        {"run_id": ""},
        {"run_id": None},
        {"step_idx": -1},
        {"step_idx": True},
        {"parallel": 1},
        {"independent": "yes"},
    ],
)
async def test_invalid_envelope_or_flags_cannot_dispatch(
    runtime_factory, tools, options
):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    with pytest.raises((ValueError, TypeError)):
        await execute(ReActToolBridge(runtime), [call(move="e2e4")], **options)
    assert effects == []
    assert await runtime.store.list_jobs() == []


@pytest.mark.parametrize("batch", [None, {}, "calls", iter(())])
async def test_invalid_batch_container(runtime_factory, tools, batch):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    with pytest.raises(TypeError):
        await execute(ReActToolBridge(runtime), batch)
    assert effects == []
    assert await runtime.store.list_jobs() == []


async def test_empty_batch_and_optional_function_type(runtime_factory, tools):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    assert await execute(bridge, []) == []
    assert await runtime.store.list_jobs() == []
    invocation = call(move="e2e4")
    del invocation["type"]
    assert payload((await execute(bridge, (invocation,)))[0])["status"] == "SUCCEEDED"
    assert effects == [("play_move", "e2e4")]


async def test_sequential_default_does_not_admit_next_call_before_success(
    runtime_factory,
):
    registry = ToolRegistry()
    entered, release = asyncio.Event(), asyncio.Event()
    order = []

    @registry.tool(idempotent=False, timeout_ms=10_000)
    async def mutate(value):
        order.append(("start", value))
        if value == 1:
            entered.set()
            await release.wait()
        order.append(("finish", value))
        return value

    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    task = asyncio.create_task(
        execute(
            bridge,
            [
                call("first", "mutate", value=1),
                call("second", "mutate", value=2),
            ],
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert len(await runtime.store.list_jobs()) == 1
        assert order == [("start", 1)]
    finally:
        release.set()
        observations = await task
    assert order == [("start", 1), ("finish", 1), ("start", 2), ("finish", 2)]
    assert [o["tool_call_id"] for o in observations] == ["first", "second"]


@pytest.mark.parametrize(
    "parallel_options",
    [
        {"parallel": True},
        {"parallel": True, "independent": True},
    ],
)
async def test_parallel_requires_declaration_and_rejects_nonidempotent_tools(
    runtime_factory,
    tools,
    parallel_options,
):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    with pytest.raises(ValueError):
        await execute(
            ReActToolBridge(runtime),
            [
                call("simulation", "simulate_move", move="e2e4"),
                call("mutation", "play_move", move="e2e4"),
            ],
            **parallel_options,
        )
    assert effects == []
    assert await runtime.store.list_jobs() == []


async def test_idempotency_alone_does_not_assert_independence(runtime_factory, tools):
    registry, effects = tools
    runtime = await runtime_factory(registry)
    with pytest.raises(ValueError, match="independent=True"):
        await execute(
            ReActToolBridge(runtime),
            [
                call("one", "simulate_move", move="e2e4"),
                call("two", "simulate_move", move="d2d4"),
            ],
            parallel=True,
        )
    assert effects == []
    assert await runtime.store.list_jobs() == []


async def test_explicit_independent_parallel_calls_overlap_and_keep_input_order(
    runtime_factory,
):
    registry = ToolRegistry()
    both_started, release_first = asyncio.Event(), asyncio.Event()
    started, finished = [], []

    @registry.tool(idempotent=True, timeout_ms=10_000)
    async def simulation(value):
        started.append(value)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        if value == 1:
            await release_first.wait()
        finished.append(value)
        return {"value": value}

    runtime = await runtime_factory(registry)
    bridge = ReActToolBridge(runtime)
    calls = [call("slow", "simulation", value=1), call("fast", "simulation", value=2)]
    task = asyncio.create_task(execute(bridge, calls, parallel=True, independent=True))
    try:
        await asyncio.wait_for(both_started.wait(), 2)
        # Observing terminal job 2 proves completion order without a timing sleep.
        jobs = await runtime.store.list_jobs()
        second = next(job for job in jobs if job.args["value"] == 2)
        await runtime.wait_for_terminal(second.job_id, timeout=2)
        assert finished == [2]
    finally:
        release_first.set()
        observations = await task
    assert finished == [2, 1]
    assert [o["tool_call_id"] for o in observations] == ["slow", "fast"]
    assert [payload(o)["result"]["value"] for o in observations] == [1, 2]
    assert await execute(bridge, calls, parallel=True, independent=True) == observations
    assert sorted(started) == [1, 2]


@pytest.mark.parametrize("failure", ["fatal", "ambiguous", "timeout"])
async def test_non_success_barrier_and_cached_quarantine_survive_restart(
    runtime_factory,
    failure,
):
    registry = ToolRegistry()
    effects = []

    @registry.tool(idempotent=False, timeout_ms=20 if failure == "timeout" else 10_000)
    async def uncertain_mutation():
        effects.append("first mutation may have happened")
        if failure == "fatal":
            raise ToolFatalError("definitive failure")
        if failure == "timeout":
            await asyncio.Event().wait()
        raise RuntimeError("response lost after possible effect")

    @registry.tool(idempotent=False)
    async def next_mutation():
        effects.append("MUST NOT RUN")
        return True

    runtime = await runtime_factory(registry)
    calls = [
        call("uncertain", "uncertain_mutation"),
        call("next", "next_mutation"),
        call("also-next", "next_mutation"),
    ]
    first = await execute(ReActToolBridge(runtime), calls)
    expected = "FAILED" if failure == "fatal" else "UNKNOWN_OUTCOME"
    failed = payload(first[0])
    assert failed["status"] == expected
    assert failed["job_id"] and failed["error"]
    assert "result" not in failed
    assert [o["tool_call_id"] for o in first] == ["uncertain", "next", "also-next"]
    for observation in first[1:]:
        blocked = payload(observation)
        assert blocked["status"] == "NOT_DISPATCHED"
        assert blocked["job_id"] is None
        assert blocked["blocked_by"] == "uncertain"
        assert blocked["error"]
    assert "NOT_DISPATCHED" not in JobStatus.__members__
    assert len(await runtime.store.list_jobs()) == 1
    await runtime.stop()
    restarted = await runtime_factory(registry, db_path=runtime.config.db_path)
    assert await execute(ReActToolBridge(restarted), calls) == first
    assert effects == ["first mutation may have happened"]
    assert len(await restarted.store.list_jobs()) == 1


async def test_idempotent_failure_also_stops_later_dependent_mutation(runtime_factory):
    registry = ToolRegistry()
    effects = []

    @registry.tool(idempotent=True, max_retries=0)
    async def failed_read():
        raise ToolFatalError("no usable input")

    @registry.tool(idempotent=False)
    async def mutation():
        effects.append("MUST NOT RUN")
        return True

    runtime = await runtime_factory(registry)
    observations = await execute(
        ReActToolBridge(runtime),
        [
            call("read", "failed_read"),
            call("write", "mutation"),
        ],
    )
    assert [payload(o)["status"] for o in observations] == ["FAILED", "NOT_DISPATCHED"]
    assert effects == []
    assert len(await runtime.store.list_jobs()) == 1


async def test_mock_demo_does_not_cross_a_cached_unknown_move_barrier(runtime_factory):
    registry = ToolRegistry()

    @registry.tool(idempotent=False)
    async def play_move(move):
        raise RuntimeError("MOCK ambiguous outcome")

    runtime = await runtime_factory(registry)
    observation = await execute(
        ReActToolBridge(runtime),
        [call("move", move="e2e4")],
        run_id="mock-react-demo-v1",
        step_idx=1,
    )
    assert payload(observation[0])["status"] == "UNKNOWN_OUTCOME"
    await runtime.stop()
    report = await run_demo(runtime.config.db_path)
    assert payload(report["first"]["one_move"][0])["status"] == "UNKNOWN_OUTCOME"
    assert "mock_shell" not in report["first"]
    assert report["effects_after_replay"] == {
        "simulate_move": 2,
        "play_move": 0,
        "execute_bash": 0,
    }
    assert report["replay_used_same_jobs"] is True


async def test_mock_demo_replays_jobs_without_extra_effects(tmp_path):
    path = tmp_path / "mock.sqlite3"
    report = await run_demo(str(path))
    assert report["mock"] is True
    assert report["replay_used_same_jobs"] is True
    assert report["extra_effects_on_replay"] is False
    assert report["first"] == report["replay"]
    assert (
        report["effects_after_first"]
        == report["effects_after_replay"]
        == {
            "simulate_move": 2,
            "play_move": 1,
            "execute_bash": 1,
        }
    )
    assert report["mock_move_history_this_process"] == ["e2e4"]
    assert list(report["first"]) == [
        "independent_simulations",
        "one_move",
        "mock_shell",
    ]
    for batch in report["first"].values():
        for observation in batch:
            result = payload(observation)
            assert result["status"] == "SUCCEEDED"
            assert result["result"]["mock"] is True
    shell = payload(report["first"]["mock_shell"][0])["result"]
    assert "nothing executed" in shell["stdout"]
    assert "NOT checkpointed" in report["caveat"]
    assert path.exists()
    second_process = await run_demo(str(path))
    assert second_process["first"] == report["first"]
    assert second_process["effects_after_first"] == {
        "simulate_move": 0,
        "play_move": 0,
        "execute_bash": 0,
    }
    assert second_process["mock_move_history_this_process"] == []


def test_cli_default_is_key_free_mock_json_with_temporary_database(
    capsys, monkeypatch, tmp_path
):
    # TemporaryDirectory's standard TMPDIR resolution is independent of the CWD.
    monkeypatch.chdir(tmp_path)
    main([])
    report = json.loads(capsys.readouterr().out)
    assert report["mock"] and report["replay_used_same_jobs"]
    assert report["extra_effects_on_replay"] is False
    assert "MOCK ONLY" in report["description"]
    assert list(tmp_path.iterdir()) == []


def test_cli_explicit_database_preserves_other_data_and_reuses_jobs(capsys, tmp_path):
    path = tmp_path / "persistent.sqlite3"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE user_data (value TEXT)")
        db.execute("INSERT INTO user_data VALUES ('keep me')")
        db.commit()
    main(["--db-path", str(path)])
    first = json.loads(capsys.readouterr().out)
    main(["--db-path", str(path)])
    replay = json.loads(capsys.readouterr().out)
    assert first["first"] == replay["first"]
    assert replay["effects_after_replay"] == {
        "simulate_move": 0,
        "play_move": 0,
        "execute_bash": 0,
    }
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT value FROM user_data").fetchall() == [("keep me",)]
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone() == (4,)
        assert db.execute("SELECT SUM(attempt_count) FROM jobs").fetchone() == (4,)
    assert Path(path).exists()
