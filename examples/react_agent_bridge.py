"""Key-free OpenAI-style tool-call bridge and local MOCK ReAct demonstration.

This is an adapter example, not an integration with or modification of an agent
framework. The original synchronous agent's sequential tools and one-move-per-
turn rule must be preserved: do not fan out stateful calls. Only explicitly
independent, idempotent calls may use parallel mode.

The caller must persist and reuse the run ID, step index, call IDs, tool names,
and arguments on replay. SQLite checkpoints tool jobs, NOT reasoning, messages,
turn boundaries, or external world state. After a non-success observation, the
caller must reconcile it before proceeding to another dependent batch/turn.

Run with ``python -m examples.react_agent_bridge``. The default database is
private and temporary. ``--db-path PATH`` retains jobs without clearing existing
data; repeated invocations reuse the fixed demo envelope and cached results.
All three demo tools are MOCKS: no chess engine, subprocess, shell, network,
model, API key, or framework is used. Mock in-memory state is not reconstructed
from cached results when reopening a database in a new process.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from runtime import (
    DurableToolRuntime,
    IdempotencyConflictError,
    JobStatus,
    RuntimeConfig,
    ToolRegistry,
)
from runtime.models import JSONValue


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    value.encode("utf-8")
    return value


def _validate_envelope(run_id: str, step_idx: int) -> None:
    _identifier(run_id, "run_id")
    if type(step_idx) is not int or step_idx < 0:
        raise ValueError("step_idx must be a nonnegative integer (not bool)")


def tool_call_key(run_id: str, step_idx: int, call_id: str) -> str:
    """SHA256 of UTF-8 compact JSON [run_id, step_idx, call_id], not joined text."""
    _validate_envelope(run_id, step_idx)
    _identifier(call_id, "call_id")
    envelope = json.dumps(
        [run_id, step_idx, call_id], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(envelope.encode("utf-8")).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _arguments(raw: object) -> tuple[dict[str, JSONValue], str]:
    if not isinstance(raw, str):
        raise TypeError("function.arguments must be a JSON string")
    try:
        args = json.loads(
            raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
        if not isinstance(args, dict):
            raise TypeError("function.arguments must decode to a JSON object")
        canonical = json.dumps(
            args,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        canonical.encode("utf-8")
    except RecursionError as exc:
        raise ValueError("function.arguments is nested too deeply") from exc
    return args, canonical


@dataclass(frozen=True)
class _Call:
    call_id: str
    name: str
    args: dict[str, JSONValue]
    args_json: str
    key: str


def _observation(call_id: str, **payload: Any) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True
        ),
    }


class ReActToolBridge:
    """Execute OpenAI-style dict calls using a caller-owned, started runtime.

    Register all tools before runtime startup/bridge construction, and do not
    change the registry afterward. Argument binding checks signatures, NOT value
    schemas. Tools remain trusted and responsible for their own value validation.

    Batches serialize within this bridge instance. Sequential mode waits for each
    terminal result before admitting the next call; any non-success blocks the
    remaining calls. NOT_DISPATCHED is an observation, never a stored JobStatus.
    At most one tool named play_move is allowed per batch; callers must also
    enforce one move per actual agent turn, including across separate batches.

    parallel=True requires independent=True as a caller assertion of independence
    AND idempotent policies for every call. Read-like names alone imply nothing.
    Results always follow input order, including in parallel mode.

    Batch validation precedes admission, including a public list_jobs() scan for
    historical key conflicts (O(stored history), suitable for this small example).
    This is not transactional batch admission: infrastructure errors, capacity,
    cancellation, or other bridge/runtime instances can race after preflight.
    Retry the SAME envelope to observe already-admitted work. Exceptions propagate
    rather than claiming such work was cancelled or safely failed. Ordering and
    failure barriers apply to this batch/instance, not to unrelated submissions.
    """

    def __init__(self, runtime: DurableToolRuntime) -> None:
        self.runtime = runtime
        self._tools = runtime.registry.snapshot()
        self._batch_lock = asyncio.Lock()

    def _prepare(
        self,
        tool_calls: Sequence[dict[str, Any]],
        run_id: str,
        step_idx: int,
        parallel: bool,
        independent: bool,
    ) -> list[_Call]:
        _validate_envelope(run_id, step_idx)
        if type(parallel) is not bool or type(independent) is not bool:
            raise TypeError("parallel and independent must be bools")
        if parallel and not independent:
            raise ValueError(
                "Parallel calls require an explicit independent=True assertion"
            )
        if not isinstance(tool_calls, (list, tuple)):
            raise TypeError("tool_calls must be a list or tuple of dicts")
        calls: list[_Call] = []
        seen: set[str] = set()
        for raw in tool_calls:
            if not isinstance(raw, dict):
                raise TypeError("Each tool call must be a dict")
            call_id = _identifier(raw.get("id"), "call_id")
            if call_id in seen:
                raise ValueError(f"Duplicate call ID in batch: {call_id!r}")
            seen.add(call_id)
            if raw.get("type", "function") != "function":
                raise ValueError("Only function tool calls are supported")
            function = raw.get("function")
            if not isinstance(function, dict):
                raise TypeError("tool call function must be a dict")
            name = _identifier(function.get("name"), "function.name")
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name!r}")
            args, canonical = _arguments(function.get("arguments"))
            tool = self._tools[name]
            tool.validate_arguments(args)
            if parallel and not tool.policy.idempotent:
                raise ValueError(f"Non-idempotent tool {name!r} cannot run in parallel")
            calls.append(
                _Call(
                    call_id,
                    name,
                    args,
                    canonical,
                    tool_call_key(run_id, step_idx, call_id),
                )
            )
        if sum(call.name == "play_move" for call in calls) > 1:
            raise ValueError("Only one play_move is allowed per batch/agent turn")
        return calls

    async def _dispatch(self, call: _Call) -> dict[str, str]:
        job_id = await self.runtime.submit(
            call.name, call.args, idempotency_key=call.key
        )
        job = await self.runtime.wait_for_terminal(job_id)
        payload: dict[str, Any] = {"job_id": job.job_id, "status": job.status.value}
        if job.status is JobStatus.SUCCEEDED:
            payload["result"] = job.result
        else:
            payload["error"] = job.error_text or f"Tool ended with {job.status.value}"
        return _observation(call.call_id, **payload)

    async def execute_tool_calls(
        self,
        tool_calls: Sequence[dict[str, Any]],
        *,
        run_id: str,
        step_idx: int,
        parallel: bool = False,
        independent: bool = False,
    ) -> list[dict[str, str]]:
        """Validate a complete batch, then return linked JSON-content tool messages.

        Malformed batches raise ValueError/TypeError without dispatching any call.
        Reusing an existing key for a different name/arguments raises
        IdempotencyConflictError (a ValueError). Same-key replays use durable jobs,
        including their original FAILED or UNKNOWN_OUTCOME terminal observations.
        """
        calls = self._prepare(tool_calls, run_id, step_idx, parallel, independent)
        if not calls:
            return []
        async with self._batch_lock:
            existing = {
                job.idempotency_key: job for job in await self.runtime.store.list_jobs()
            }
            for call in calls:
                job = existing.get(call.key)
                if job is not None and (
                    job.tool_name != call.name or job.args_json != call.args_json
                ):
                    raise IdempotencyConflictError(
                        f"Call {call.call_id!r} already identifies a different request"
                    )
            if parallel:
                return list(
                    await asyncio.gather(*(self._dispatch(call) for call in calls))
                )
            observations: list[dict[str, str]] = []
            blocked_by: str | None = None
            for call in calls:
                if blocked_by is not None:
                    observations.append(
                        _observation(
                            call.call_id,
                            job_id=None,
                            status="NOT_DISPATCHED",
                            error="Earlier sequential call did not succeed; reconcile before proceeding",
                            blocked_by=blocked_by,
                        )
                    )
                    continue
                observation = await self._dispatch(call)
                observations.append(observation)
                if (
                    json.loads(observation["content"])["status"]
                    != JobStatus.SUCCEEDED.value
                ):
                    blocked_by = call.call_id
            return observations


def _mock_registry() -> tuple[ToolRegistry, dict[str, int], list[str]]:
    """Build MOCK tools and process-local counters, not a chess/shell environment."""
    registry = ToolRegistry()
    effects = {"simulate_move": 0, "play_move": 0, "execute_bash": 0}
    moves: list[str] = []

    @registry.tool(idempotent=True)
    async def simulate_move(move: str) -> JSONValue:
        """MOCK independent simulation: a canned label, not chess analysis."""
        effects["simulate_move"] += 1
        await asyncio.sleep(0)
        return {
            "mock": True,
            "candidate": move,
            "evaluation": "MOCK ONLY; no chess engine",
        }

    @registry.tool(idempotent=False)
    async def play_move(move: str) -> JSONValue:
        """MOCK stateful move: append text once; no legality checks or chess engine."""
        effects["play_move"] += 1
        moves.append(move)
        return {"mock": True, "move": move, "mock_move_number": len(moves)}

    @registry.tool(idempotent=False)
    async def execute_bash(command: str) -> JSONValue:
        """MOCK ONLY: record command text; NEVER execute code, shell, or subprocesses."""
        effects["execute_bash"] += 1
        return {
            "mock": True,
            "command": command,
            "stdout": "MOCK ONLY: command recorded as text; nothing executed.",
        }

    return registry, effects, moves


def _demo_call(call_id: str, name: str, **args: JSONValue) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


async def run_demo(db_path: str) -> dict[str, Any]:
    """Run and replay a fixed MOCK envelope; never delete/reset an existing DB."""
    registry, effects, moves = _mock_registry()
    config = RuntimeConfig(db_path=db_path, max_workers=2)
    async with DurableToolRuntime(config, registry) as runtime:
        bridge = ReActToolBridge(runtime)
        batches = [
            (
                "independent_simulations",
                [
                    _demo_call("simulation-a", "simulate_move", move="e2e4"),
                    _demo_call("simulation-b", "simulate_move", move="d2d4"),
                ],
                True,
            ),
            ("one_move", [_demo_call("move", "play_move", move="e2e4")], False),
            (
                "mock_shell",
                [_demo_call("shell", "execute_bash", command="echo MOCK_ONLY")],
                False,
            ),
        ]

        async def execute_envelope() -> dict[str, list[dict[str, str]]]:
            observations = {}
            for step_idx, (label, calls, parallel) in enumerate(batches):
                observations[label] = await bridge.execute_tool_calls(
                    calls,
                    run_id="mock-react-demo-v1",
                    step_idx=step_idx,
                    parallel=parallel,
                    independent=parallel,
                )
                if any(
                    json.loads(item["content"])["status"] != JobStatus.SUCCEEDED.value
                    for item in observations[label]
                ):
                    break
            return observations

        first = await execute_envelope()
        effects_after_first = effects.copy()
        replay = await execute_envelope()
        return {
            "mock": True,
            "description": "MOCK ONLY: no model, network, shell execution, or chess engine",
            "first": first,
            "replay": replay,
            "replay_used_same_jobs": first == replay,
            "effects_after_first": effects_after_first,
            "effects_after_replay": effects.copy(),
            "extra_effects_on_replay": effects != effects_after_first,
            "mock_move_history_this_process": moves.copy(),
            "caveat": "Tool jobs only: reasoning and mock world state are NOT checkpointed. "
            "An existing DB can return cached results with zero effects in this process.",
        }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", type=Path, help="Persistent SQLite file; existing jobs are kept"
    )
    options = parser.parse_args(argv)
    if options.db_path is not None:
        report = asyncio.run(run_demo(str(options.db_path)))
    else:
        with TemporaryDirectory(prefix="react-bridge-mock-") as directory:
            report = asyncio.run(run_demo(str(Path(directory) / "jobs.sqlite3")))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
