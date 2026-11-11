# Durable Tool-Execution Runtime

A Python execution backend for AI-agent tool calls, built with `asyncio` and SQLite WAL. It persists jobs before acknowledging them, executes registered async tools with bounded concurrency, and recovers unfinished work after process crashes.

The runtime handles **tool execution, not model reasoning**. It is model-agnostic and requires no API keys, external broker, or network services to run its examples and tests.

## What it does

- Deduplicates submissions using an idempotency key and a fingerprint of the tool name and JSON arguments.
- Bounds unfinished jobs and concurrent executions, rejecting new work when capacity is exhausted.
- Uses heartbeat-renewed leases and attempt-based fencing to reject stale result writes.
- Retries eligible idempotent calls and quarantines uncertain non-idempotent calls as `UNKNOWN_OUTCOME`.
- Persists state transitions and audit events atomically.
- Includes a real `SIGKILL` recovery harness, reproducible latency benchmarks, and an OpenAI-style tool-call adapter.

## Quickstart

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/soumomo/durable-tool-execution.git
cd durable-tool-execution
uv sync --locked
uv run python -m examples.react_agent_bridge
uv run pytest -q
```

The example executes mock tools, replays the same calls, and reports `replay_used_same_jobs: true` and `extra_effects_on_replay: false`. It does not call an LLM or execute shell commands.

All subsequent commands run from the cloned repository's root. The Python distribution is named `durable-tool-runtime` and exports the `runtime` package; examples and benchmarks run from the checkout. `.python-version` selects Python 3.11, and `uv.lock` pins development dependencies. There are no third-party runtime dependencies.

## Usage

Register trusted async functions with explicit execution policies. This complete example stores two document lookups in `jobs.sqlite3`; running it again with the same keys returns the existing completed jobs.

```python
import asyncio

from runtime import DurableToolRuntime, RuntimeConfig, ToolRegistry

registry = ToolRegistry()


@registry.tool(idempotent=True, timeout_ms=500, max_retries=2)
async def fetch_document(document_id: str):
    await asyncio.sleep(0)
    return {"document_id": document_id, "text": "Example document"}


async def main():
    config = RuntimeConfig(db_path="jobs.sqlite3", max_workers=4)
    async with DurableToolRuntime(config, registry) as runtime:
        job_ids = await asyncio.gather(*(
            runtime.submit(
                "fetch_document", {"document_id": document_id},
                idempotency_key=f"demo:{document_id}",
            )
            for document_id in ("doc-1", "doc-2")
        ))
        results = await asyncio.gather(*(
            runtime.wait_for_terminal(job_id, timeout=2) for job_id in job_ids
        ))
        for job in results:
            print(job.status.value, job.result)


asyncio.run(main())
```

`max_retries=2` permits **three total attempts**. Mark a tool idempotent only if replay is actually safe, either by its behavior or through downstream idempotency support.

## Architecture

```text
Admission: validate → deduplicate → check capacity → insert job + audit event → commit → acknowledge

Execution: PENDING → LEASED (attempt N) → SUCCEEDED
                                      → RETRY_WAIT → LEASED (attempt N+1)
                                      → FAILED
                                      → UNKNOWN_OUTCOME
```

SQLite is the source of truth. A dedicated single-thread executor owns the connection; SQLite operations do not block the asyncio event-loop thread. Admission, claims, outcomes, and their audit events use explicit `BEGIN IMMEDIATE` transactions.

A fixed worker pool pulls eligible jobs from SQLite. Notifications reduce idle latency, and polling discovers committed work even if a notification is missed. There is no second in-memory job queue to reconcile after a crash. Startup and periodic recovery sweeps inspect expired leases; unexpired leases remain untouched until expiry.

### Correctness guarantees

| Guarantee | Behavior |
| --- | --- |
| Commit before acknowledgement | A new job ID is returned only after the job and admission event commit. If the response is lost, resubmitting the same key finds the committed job. |
| Deduplication before backpressure | The same key, tool, and canonical JSON arguments return the existing job, even at capacity. Different tools or arguments raise `IdempotencyConflictError`. |
| Bounded unfinished backlog | `max_outstanding_jobs` counts `PENDING`, `LEASED`, and `RETRY_WAIT` inside the admission transaction. New jobs at capacity raise `BackpressureError` without a write. |
| Fenced leases | Each claim increments `attempt_count`. Heartbeats and outcomes require a matching attempt, `LEASED` status, and an unexpired lease. Stale outcomes leave job state unchanged and record `STALE_COMPLETION_FENCED`. |
| Safe recovery policy | Expired idempotent calls retry within their persisted budget. Uncertain non-idempotent calls become terminal `UNKNOWN_OUTCOME` for reconciliation, without automatic replay. |
| Atomic audit history | State changes and their events commit or roll back together. Event IDs define ordering; heartbeats do not add state-transition events. |

**Fencing protects stored results, not external side effects.** A worker may reach a remote API before losing its lease. Safe replay depends on actual tool semantics; this runtime does not guarantee exactly-once external effects.

Arguments must be strict JSON objects, and results must be JSON values. Non-string object keys, non-finite numbers, tuples, arbitrary objects, and cycles are rejected. Object keys are sorted for stable fingerprints; `1` and `1.0` retain distinct representations.

### Outcomes and retries

| Outcome | Idempotent tool | Non-idempotent tool |
| --- | --- | --- |
| Expired lease, timeout, `ToolRetryableError`, or tool self-cancellation | Retry if budget remains; otherwise `FAILED` | `UNKNOWN_OUTCOME` |
| `ToolFatalError` indicating a known definitive failure | `FAILED` | `FAILED` |
| Unexpected tool exception or invalid JSON result | `FAILED` | `UNKNOWN_OUTCOME` |

Retries use full-jitter exponential backoff, capped at 60 seconds. Only claims increment the attempt counter. Each job retains its admitted timeout and retry policy across restarts; persisted policies do not snapshot function code, default argument values, or remote state.

Background storage and scheduler failures are infrastructure failures, not tool failures. They stop admission, cancel peer workers, and retain the underlying cause for shutdown to surface. Storage errors encountered directly by a submission or observer propagate to that caller. Deadlines cancel the tool once and wait for cleanup before recording an outcome; cleanup may extend beyond the execution timeout.

### Lifecycle

- Register all tools before startup. `start()` snapshots registry membership and validates unfinished jobs against available tool signatures and idempotency policies before dispatching work. Signature binding checks argument names, not annotation-based value schemas.
- Use the async context manager, or explicitly call `start()` and `stop()`. `stop(graceful=True)` rejects new submissions and drains accepted work, including retries.
- `stop(graceful=False)` cancels active calls and leaves unfinished leases for recovery. It does not declare interrupted side effects safe to repeat. A stopped instance is one-shot; restart with a new instance.
- Cancelling graceful shutdown switches to immediate cleanup. Owned tasks and storage are joined before cancellation propagates, including repeated cancellation. Concurrent stop calls serialize.
- `wait_for_terminal(job_id, timeout=None)` observes a job. Cancelling or timing out an observer does not cancel execution or other observers. Unknown IDs raise `KeyError`.

## Tool-call adapter

The included [adapter and runnable demo](examples/react_agent_bridge.py) accept OpenAI-style tool-call dictionaries and return linked `role="tool"` messages containing the durable job ID, terminal status, and result or error. No agent framework or model SDK is required.

```sh
uv run python -m examples.react_agent_bridge
# Retain jobs and replay the same demo calls across invocations:
uv run python -m examples.react_agent_bridge --db-path mock-jobs.sqlite3
```

The demo defines three mock tools: `simulate_move`, `play_move`, and `execute_bash`. Simulations run independently in parallel; a stateful move and a mock command run sequentially. There is no chess engine, shell execution, or network access. Reopening a database reuses durable results but does not reconstruct process-local mock state.

For integration with an agent loop:

- Construct `ReActToolBridge` with a started runtime and call `execute_tool_calls(tool_calls, run_id=..., step_idx=...)`. Register tools before startup and do not mutate the registry afterward.
- Persist and reuse run, step, and call identities on replay. Keys hash structured JSON `[run_id, step_idx, call_id]`, avoiding ambiguous delimiter concatenation.
- Sequential execution is the default. A non-success blocks remaining dependent calls with `NOT_DISPATCHED` observations; that is not a stored job status.
- Parallel execution requires `parallel=True`, `independent=True`, and idempotent tool policies. Idempotency does not itself establish independence.
- The adapter permits one `play_move` per batch. Callers enforce ordering, turn boundaries, and reconciliation across separate batches or bridge instances.
- Batch validation precedes admission, including a scan for historical key conflicts. Admission is not transactional across the batch, and the history scan is intended for this example's scale.

The adapter persists **tool jobs**, not model reasoning, messages, full workflows, or external world state.

## Process-crash verification

The [chaos harness](benchmarks/chaos_crash_test.py) waits for an observed crash point: 40 completed jobs, eight leased calls with durable mock effects, and 52 pending jobs. It sends `SIGKILL` to its owned child, reaps it, and starts a fresh process against the same database. An independent invocation/effect ledger checks replay behavior alongside job states and audit paths.

```sh
uv run python -m benchmarks.chaos_crash_test
uv run python -m benchmarks.chaos_crash_test --synchronous NORMAL
```

Recorded `FULL` output:

```text
SIGKILL chaos: SQLite synchronous=FULL
Crash point: accepted=100, SUCCEEDED=40, LEASED=8 (4 idempotent + 4 non-idempotent), PENDING=52
Child exit: -9 (SIGKILL); recovery exit: 0
Recovery: SUCCEEDED=88, FAILED=8, UNKNOWN_OUTCOME=4, unfinished=0
Conservation: 100 accepted = 88 succeeded + 8 failed + 4 unknown (PASS)
Precompleted: 40 unchanged; 0 reexecutions
Non-idempotent blockers: 4 effects before crash; 0 UNKNOWN_OUTCOME replays
Idempotent blockers: 8 invocations; 4 deduplicated effects
Fail-once tools: 40 invocations; 20 effects
SQLite runtime + ledger: integrity_check=ok; foreign_key_check=clean
Post-SIGKILL WAL observed before reopening: yes
Verification checks: 24/24 (PASS)
Elapsed: 2.188 s
```

The eight failures are intentional definitive tool rejections. Idempotent mock effects deduplicate downstream; their replay demonstrates the tool contract, not runtime exactly-once execution. Both durability modes produced the same state counts: [FULL report](benchmarks/results/chaos-full.json), [NORMAL report](benchmarks/results/chaos-normal.json).

The harness uses temporary databases and kills only processes it creates. It requires POSIX `SIGKILL` and has been verified on macOS, not Linux. Its timeout is cooperative: joining a spawn or offloaded database operation may exceed the deadline before bounded child reaping completes. A process-kill test does **not** prove power-loss durability.

## Benchmarks

Recorded on **2026-11-11**: Apple M4, 10 logical CPUs, arm64, macOS 27.2, CPython 3.11.13, SQLite 3.51.2. Each mode uses a fresh temporary database, 100 excluded warmup calls, and 1,000 measured no-op calls with 16 workers and 16 closed-loop clients. Each client submits a call and observes its terminal result before submitting another.

```sh
uv run python -m benchmarks.bench_latency
```

| Mode | Jobs | Workers / clients | Elapsed (s) | Jobs/sec |
| --- | ---: | ---: | ---: | ---: |
| FULL | 1,000 | 16 / 16 | 0.473 | 2,114.7 |
| NORMAL | 1,000 | 16 / 16 | 0.456 | 2,195.2 |

| Mode | Interval | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | --- | ---: | ---: | ---: |
| FULL | Admission: submit → acknowledgement | 2.182 | 3.247 | 3.629 |
| FULL | Admission: submit → DB post-commit hook | 2.146 | 3.203 | 3.591 |
| FULL | Queue + dispatch: DB post-commit hook → tool start | 0.968 | 1.694 | 1.986 |
| FULL | Completion: tool return → DB post-commit hook | 2.666 | 3.657 | 4.028 |
| FULL | End-to-end: submit → terminal observation | 7.467 | 8.028 | 8.667 |
| NORMAL | Admission: submit → acknowledgement | 1.981 | 5.441 | 6.108 |
| NORMAL | Admission: submit → DB post-commit hook | 1.945 | 5.402 | 6.075 |
| NORMAL | Queue + dispatch: DB post-commit hook → tool start | 0.463 | 1.300 | 3.901 |
| NORMAL | Completion: tool return → DB post-commit hook | 2.331 | 5.724 | 15.145 |
| NORMAL | End-to-end: submit → terminal observation | 6.078 | 10.003 | 19.113 |

[Recorded measurements and environment metadata](benchmarks/results/latency.json) include full precision. These are single-machine, single-run no-op results, not production SLAs or an overload test. `NORMAL` achieved slightly higher throughput but worse p99 in this run. Every measured job succeeded on its first attempt with the expected result.

### Measurement boundaries

- Intervals use `perf_counter_ns()`. Percentiles use linear interpolation at `(n - 1) * q`.
- Benchmark-only hooks sample after transactional store methods return on the database thread, after commit. A worker may start before `submit()` returns, so dispatch timing starts at this hook—not at caller acknowledgement. Missing or out-of-order timestamps fail validation.
- Timings include instrumentation, scheduling, serialization, executor queueing, and result-observer queries. Completion timing is not raw `fsync` latency. Throughput excludes startup, warmup, and shutdown but includes result observation.
- No model, network, retry, or external-effect latency is included. Memory limits, prolonged load, and remote-service throughput are not measured.

To record another run, add `--json PATH`. Profile separately from baseline measurements:

```sh
uv run python -m benchmarks.bench_latency --profile benchmarks/results/latency.prof
uv run python -c 'import pstats; pstats.Stats("benchmarks/results/latency.prof").strip_dirs().sort_stats("cumulative").print_stats(20)'
```

The profiler covers the event-loop thread, not Python execution inside the SQLite executor thread. Profiled timings are not baseline results; `.prof` files are ignored by Git.

## Tests and build

```sh
uv run pytest -q
uv run pytest -m acceptance -q
uv run pytest -m chaos -q
PYTHONASYNCIODEBUG=1 uv run pytest -q
uv run --isolated --python 3.14 pytest -q
uv build
```

Recorded verification on macOS:

| Environment | Result |
| --- | --- |
| Python 3.11.13 | 560 tests passed |
| Python 3.11.13, asyncio debug enabled | 560 tests passed |
| Python 3.14.2, isolated environment | 560 tests passed |

The `acceptance` marker selects 10 cases covering deduplication at capacity, bounded backlog/concurrency, stale-worker fencing, timeout policies, committed-job discovery after restart, and audit chronology. The `chaos` marker selects subprocess checks. The full suite also covers lifecycle/cancellation races, infrastructure failures, bridge replay, and benchmark/profile smoke tests. Pytest does not enforce performance thresholds.

## Operating limits

- **Single-process scheduling.** Independent-connection atomic claims are tested, but distributed scheduling is not supported. Use a local filesystem suitable for SQLite WAL, not a shared network database file. `busy_timeout=5000` permits bounded waiting for other writers; the dedicated executor does not eliminate external contention.
- **Durability depends on the storage stack.** The default is WAL with `synchronous=FULL`. `NORMAL` can lose recent commits after an OS crash or power failure. `FULL` strengthens durability subject to SQLite, filesystem, hardware, and platform sync guarantees; neither mode replaces backups.
- **Leases use wall time.** Large clock adjustments can affect expiration. Distributed-clock tolerance is not provided.
- **Tools are trusted and cooperative.** Timeouts cannot forcibly stop blocking code, undo remote operations, or guarantee bounded cleanup. Tools own any additional tasks they spawn. This is not a code sandbox.
- **Capacity is not a total memory/disk bound.** Argument sizes, concurrent submission callers, terminal history, and audit retention need separate service-level limits. History pruning is not implemented.
- **Execution durability is not workflow checkpointing.** Reconciliation of `UNKNOWN_OUTCOME`, agent state, and downstream effects remains an application concern. The runtime does not provide exactly-once external effects.

Workers pull directly from SQLite, and each transaction commits independently. Custom group commit, per-tool rate limiting, circuit breakers, HTTP serving, and distributed coordination are outside the current scope.

## Repository layout

```text
runtime/                  # Models, registry, SQLite store, and worker engine
examples/                 # Standalone tool-call adapter and mock demo
benchmarks/               # SIGKILL recovery harness and latency/profiling CLI
benchmarks/results/       # Recorded aggregate benchmark and recovery reports
tests/                    # Store, runtime, adapter, and harness tests
pyproject.toml            # Package metadata and test configuration
uv.lock                   # Pinned development dependencies
```
