# Durable Tool-Execution Runtime

A deterministic execution backend for registered, trusted async agent tools. Fixed workers execute calls with deadlines and heartbeat-renewed leases; SQLite WAL is the source of truth for admission, results, retries, and restart recovery. No external broker is required.

**All five build phases are complete:** durable store, worker engine, acceptance suite, real process-kill recovery, measured benchmarks, and a key-free ReAct-format bridge. This is execution infrastructure—not a model-powered product, secure code sandbox, or distributed workflow engine.

## Development

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required for these commands:

```sh
uv sync
uv run pytest -m acceptance -q  # Focused six-criterion verification gate
uv run pytest -q
uv run python -m examples.react_agent_bridge
uv run python -m benchmarks.chaos_crash_test
uv run python -m benchmarks.bench_latency
```

There are no runtime dependencies or required API keys. Pytest and pytest-asyncio are development dependencies. `.python-version` selects Python 3.11; `uv.lock` records the resolved environment. Run the commands above from this repository's root: the wheel packages `runtime`, while examples and benchmark tools live in the checkout. The chaos harness requires POSIX `SIGKILL` (verified on macOS; not tested on Linux).

### Verification

| Environment | Result |
| --- | --- |
| Python 3.11.13 | 560 tests passed |
| Python 3.11.13, `PYTHONASYNCIODEBUG=1` | 560 tests passed |
| Python 3.14.2, isolated environment | 560 tests passed |

Coverage includes store atomicity, concurrent admission/claims, bounded execution, timeout classification, heartbeat renewal through result persistence, startup/periodic recovery, waiter isolation, and repeated shutdown cancellation. Regression tests cover deadlines interrupting tool cleanup and infrastructure errors racing shutdown. The suite also runs actual subprocess-kill checks, bridge validation/replay, and small benchmark/profile smoke tests. There are no timing-performance thresholds in pytest.

### Acceptance map

`uv run pytest -m acceptance -q` selects **10 cases within the full 560-test suite**, covering the six criteria below. This focused gate excludes process-kill tests and benchmarks; `uv run pytest -m chaos -q` selects the subprocess chaos checks.

| Acceptance criterion | Coverage |
| --- | --- |
| 1. Deduplication and conflicts before backpressure; one execution per logical job | [`test_fixed_worker_bound_backlog_and_duplicate_admission_at_capacity`](tests/test_runtime.py): duplicate requests succeed while full; changed arguments conflict; new keys are rejected; each accepted call executes once. |
| 2. Bounded outstanding backlog and worker concurrency | The same capacity test checks two active workers, three pending jobs, a five-job limit, and capacity reuse after completion. |
| 3. A expires, B commits, A finishes late without overwriting B | [`test_fenced_lease_prevents_stale_worker_overwrite`](tests/test_acceptance_fencing.py), under both `FULL` and `NORMAL`: checks the real old epoch, the winning record, fencing counter, and durable audit event. |
| 4. Timeouts and expired leases follow the tool's idempotency policy | Timeout cases in [`test_tool_outcome_classification_and_safe_replay`](tests/test_runtime.py), plus `LEASED` cases in [`test_restart_runs_persisted_work_and_recovers_expired_leases`](tests/test_runtime_lifecycle.py): safe replay versus `UNKNOWN_OUTCOME` without another tool call. |
| 5. Committed work is discoverable without a dispatch notification | `PENDING` cases in the restart test: a passive store commits and closes without running a worker; a fresh runtime discovers and executes that row. |
| 6. Exact audit chronology across a retry | [`test_audit_log_exact_chronology_includes_attempts_but_not_heartbeats`](tests/test_store.py) checks transition pairs, attempt numbers, timestamps, and event order. Runtime outcome tests additionally verify the full transition chain and returned result. |

The live fencing test pauses only A's heartbeat and advances a controlled database clock. Real worker tasks, the periodic recovery sweep, atomic claims, and fenced result writes remain in use; A is released only after B's success is persisted. There are no sleep-based race-ordering assumptions or manufactured attempt tokens.

The commit-before-dispatch test models that persistence boundary; **it does not kill a process**. The separate chaos harness below does. Neither test proves power-loss safety.

## Architecture

```text
admission: validate → deduplicate → check capacity → insert job + event → commit → return ID
execution: PENDING → LEASED (attempt N) → SUCCEEDED
                                      → RETRY_WAIT → LEASED (attempt N+1)
                                      → FAILED
                                      → UNKNOWN_OUTCOME
```

One `sqlite3.Connection` is created, used, and closed on a dedicated single-thread executor. No SQLite operations or executor shutdown waits run on the asyncio event-loop thread. Admission, claiming, outcomes, and their audit events use explicit `BEGIN IMMEDIATE` transactions.

The runtime targets one process with a fixed number of async workers. Atomic claims are tested across independent connections, but that is not a claim of distributed scheduling support. Workers pull from SQLite; there is no second job queue to reconcile. Rotating event notifications avoid lost wakeups, and periodic polling discovers committed work even if a notification never arrives.

## Runtime example

This example uses local mock document data: no model API, network service, or key is needed. Re-running it returns the original completed jobs for the same keys.

```python
import asyncio

from runtime import DurableToolRuntime, RuntimeConfig, ToolRegistry

registry = ToolRegistry()


@registry.tool(idempotent=True, timeout_ms=500, max_retries=2)
async def fetch_document(document_id: str):
    await asyncio.sleep(0)
    return {"document_id": document_id, "text": "Local demo document"}


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

### Registry and lifecycle

- `registry.register(policy, async_function)` registers a trusted tool; `@registry.tool(...)` is the decorator equivalent. Duplicate names, synchronous functions, and required positional-only parameters are rejected. Signature binding validates argument names, not annotation-based value schemas.
- `await runtime.start()` snapshots registry membership, validates unfinished jobs against available tools, recovers expired leases, then starts `max_workers` workers and one reaper. Missing tools or incompatible idempotency/signatures fail startup without dispatching work. Terminal historical jobs need not be registered.
- `await runtime.submit(name, args, idempotency_key)` returns the durable job ID. Accepted jobs use their original persisted policy even after registry timeout/retry defaults change. Policies do not snapshot tool code, default argument values, or remote state: deployments must keep implementations compatible.
- `await runtime.wait_for_terminal(job_id, timeout=None)` observes completion. Its timeout or cancellation does not cancel execution or another observer. Unknown IDs raise `KeyError`.
- `await runtime.stop(graceful=True)` rejects new submissions and drains accepted work, including retries. `graceful=False` cancels active calls and leaves unfinished leases for a later runtime to reconcile; it does not pretend interrupted external actions failed safely. A stopped instance is one-shot; use a new instance for restart.
- Cancelling graceful shutdown switches to immediate cleanup. Cancellation propagates only after owned tasks and the database have been cleaned up, even if cancellation is repeated. Concurrent stop calls serialize; a second call does not escalate an already-draining shutdown.

Defaults are a 50 ms idle/result polling interval and a 100 ms lease-recovery interval, configurable with `poll_interval_ms` and `recovery_interval_ms`. Heartbeats use `heartbeat_interval_ms`; deadlines and retry budgets come from each job's persisted `ToolPolicy`.

## Contracts and guarantees

- **Commit before acknowledgement.** Returning a new job means its `PENDING` row and admission event have committed. A crash after commit but before the caller receives a response is ambiguous to the caller; resubmit with the same key. Cancelling an awaiting coroutine does not prove the database transaction was aborted.
- **Idempotency before backpressure.** The same key, tool name, and canonical JSON arguments return the original job, even at capacity. Different tools or arguments raise `IdempotencyConflictError`. Policy defaults are not part of request identity: an existing job retains its original persisted policy.
- **Strict JSON.** Arguments must be JSON objects; results may be any JSON value. Non-string object keys, non-finite numbers, tuples, arbitrary objects, and cycles are rejected. Objects are key-sorted for stable fingerprints; values such as `1` and `1.0` retain distinct representations.
- **Bounded unfinished backlog.** `max_outstanding_jobs` counts `PENDING`, `LEASED`, and `RETRY_WAIT` jobs inside the admission transaction. New submissions at capacity raise `BackpressureError` without writing a job. Terminal jobs remain available for deduplication and inspection; retention/pruning is not implemented.
- **Fenced, expiring leases.** Each claim increments `attempt_count`. Heartbeats and outcomes require a matching attempt, `LEASED` status, and `lease_expires_at > now`. An expired lease cannot be revived before the recovery sweep. Stale outcome writes leave job state unchanged and commit a `STALE_COMPLETION_FENCED` audit event before raising `StaleLeaseError`.
- **Bounded retries.** `max_retries=2` allows three total attempts. Only claims increment the attempt counter. Idempotent transient failures and expired leases retry with full-jitter exponential backoff, capped at 60 seconds. Exhaustion becomes `FAILED`; retry eligibility does not promise eventual success.
- **Ambiguous side effects are quarantined.** An uncertain non-idempotent outcome becomes terminal `UNKNOWN_OUTCOME`, regardless of retry budget. A caller may pass `outcome_unknown=False` only when a failure is known to be definitive; this becomes `FAILED` for a non-idempotent tool. The store never automatically replays non-idempotent tools.
- **Audit atomicity.** Each state transition and its event commit or roll back together. Heartbeats renew a lease without adding a state-transition event. Event IDs, not wall-clock timestamps, define audit ordering.
- **Automatic runtime recovery.** Runtime startup and a periodic reaper reconcile expired leases; future-dated leases are left alone until expiry. The lower-level `SQLiteJobStore` remains passive: opening it alone does not start workers or recovery tasks.
- **Cancellation-safe cleanup.** Use the async context manager or explicitly await `runtime.stop()` / `store.close()`. Startup failure and shutdown finish connection cleanup on the owning thread before propagating cancellation, including repeated cancellation. Cleanup can wait for already-running database operations; it does not forcibly interrupt them.

Fencing protects the stored result, **not external side effects**. A worker that has lost its lease may still have reached an external API. Safe replay depends on the tool's real idempotency properties or downstream idempotency support. This is not an exactly-once guarantee.

### Tool outcomes and infrastructure failures

| Outcome | Idempotent tool | Non-idempotent tool |
| --- | --- | --- |
| Timeout, `ToolRetryableError`, or tool self-cancellation | Retry if budget remains; otherwise `FAILED` | `UNKNOWN_OUTCOME` |
| `ToolFatalError` (known definitive failure) | `FAILED` | `FAILED` |
| Unexpected tool exception or invalid JSON result | `FAILED` | `UNKNOWN_OUTCOME` |

An expired lease follows the same safe-replay policy. The runtime cancels a call when its heartbeat loses ownership, while recovery reconciles the durable state. Stale writes are fenced without killing the worker. Per-instance `lost_leases` and `fenced_stale_completions` counters expose those paths.

Storage/scheduler failures are not tool errors. They latch the original cause immediately, reject further submissions and observations with chained `RuntimeError`, and cancel peer workers. Shutdown cleans up before surfacing the failure; if the shutdown caller is also cancelled, cancellation propagates while the original infrastructure failure remains retained. Use the context manager or explicitly stop a failed runtime to close its store.

Deadlines are controlled outside the actual tool task. On timeout or shutdown, the runtime cancels the tool once and joins its cleanup before recording an outcome or closing storage; the execution deadline cannot issue a second cancellation into an async `finally` block. Cleanup can extend beyond the configured execution timeout.

## Durability and operating limits

The default is `journal_mode=WAL` with `synchronous=FULL`. `NORMAL` is configurable but may lose recent commits after an OS crash or power failure. `FULL` strengthens durability subject to SQLite, filesystem, hardware, and platform sync guarantees; neither mode substitutes for backups or proves power-loss safety on every device. The process-kill harness tests process failure, not power failure.

`busy_timeout=5000` allows bounded waiting for another SQLite writer. A single executor serializes this store's operations; it does not eliminate contention from other connections or processes. Use a local filesystem suitable for SQLite WAL, not a shared network database file.

Persisted lease timestamps use Unix wall time so they survive process restarts. Large clock adjustments can affect expiration; this release does not claim distributed-clock tolerance. Synthetic `now=` timestamps and an injectable random source support deterministic store tests; production callers should omit `now`.

The unfinished-job limit does not bound argument sizes, terminal history, or arbitrary numbers of concurrent callers waiting to submit. An eventual service wrapper needs its own ingress limits. Trusted tools must cooperate with asyncio and finish cancellation cleanup; timeouts cannot forcibly stop blocking code, undo remote operations, or guarantee a fixed shutdown duration. Tools are responsible for any additional tasks they spawn.

## Real process-kill recovery

```sh
uv run python -m benchmarks.chaos_crash_test --json benchmarks/results/chaos-full.json
uv run python -m benchmarks.chaos_crash_test --synchronous NORMAL --json benchmarks/results/chaos-normal.json
```

The parent waits for **observed state**, not an arbitrary sleep: 40 completed jobs, eight leased calls with durable mock side effects, and 52 pending jobs. It sends real `SIGKILL` to its owned child, reaps it, and starts a fresh process against the same database. An independent effect/invocation ledger checks replay behavior in addition to SQLite job counts and audit paths. Idempotent mock effects deduplicate downstream; this is not a runtime exactly-once claim.

Verbatim output from the saved `FULL` run:

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

The eight `FAILED` jobs are intentional definitive tool rejections, not lost work. Both `FULL` and `NORMAL` produced the same verified state counts. Saved reports: [FULL](benchmarks/results/chaos-full.json), [NORMAL](benchmarks/results/chaos-normal.json). Temporary databases are removed; the harness never accepts a user database path and kills only children it creates. Its timeout is cooperative: safely joining a spawn or offloaded file/database operation can exceed the deadline, followed by bounded child reaping.

## Measured local performance

```sh
uv run python -m benchmarks.bench_latency --json benchmarks/results/latency.json
```

One unprofiled run on **Apple M4, 10 logical CPUs, arm64, macOS 27.2, CPython 3.11.13, SQLite 3.51.2**. Each mode uses a fresh temporary database, 100 warmup calls excluded from measurements, then 1,000 no-op calls with 16 workers and 16 closed-loop clients. Each client submits, observes its terminal result, then submits again; this is not an open-loop saturation or overload test. Every measured job succeeded on its first attempt with the expected result.

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

[Raw aggregate measurements and environment metadata](benchmarks/results/latency.json) include full precision. These are single-machine, single-run no-op measurements—not production SLAs, HFT claims, or evidence that `NORMAL` always improves latency. In this run it improved aggregate throughput slightly but worsened p99. Repeat on your target hardware before drawing conclusions.

### Measurement boundaries and profiling

- All intervals use `perf_counter_ns()`, not persisted wall-clock timestamps. Percentiles use linear interpolation at `(n - 1) * q`.
- Benchmark-only store hooks timestamp immediately after transactional admission/outcome methods return on the database thread, **after commit**. They do not instrument the production runtime.
- A worker can start before `submit()` returns to the caller. Queue/dispatch timing therefore starts at the DB hook, **not the caller's acknowledgement**. Missing or out-of-order timing fails the benchmark instead of being clamped.
- Timings include instrumentation, event-loop scheduling, serialization, executor queueing, and result-observer queries. Completion timing is not raw `fsync` latency. Throughput excludes startup, warmup, and shutdown, but includes clients observing every result.
- This workload has no model/network latency, retries, or external side effects. It does not measure memory limits, prolonged load, or remote-service throughput. Backpressure and failures have separate correctness tests.

Profile separately; do not use profiled timings as the baseline:

```sh
uv run python -m benchmarks.bench_latency --profile benchmarks/results/latency.prof
uv run python -c 'import pstats; pstats.Stats("benchmarks/results/latency.prof").strip_dirs().sort_stats("cumulative").print_stats(20)'
```

The profiler covers the calling/event-loop thread, not Python execution inside the dedicated SQLite thread. Its output is useful for scheduling and executor overhead, not a breakdown of database sync time. Binary `.prof` files are ignored; aggregate baseline JSON is retained.

## ReAct-format bridge

[`examples/react_agent_bridge.py`](examples/react_agent_bridge.py) adapts OpenAI-style **dict** tool-call envelopes without importing the original ReAct framework, Modal, SWE-ReX, a model SDK, or a chess engine. All demo tools—including `execute_bash`—are mocks; no shell command is executed.

```sh
uv run python -m examples.react_agent_bridge
# Optional: retain results and replay the same fixed demo envelope across runs.
uv run python -m examples.react_agent_bridge --db-path mock-jobs.sqlite3
```

The demo runs two independent mock simulations in parallel, then one stateful mock move and a mock command sequentially. It replays the same envelope and reports `replay_used_same_jobs: true` and `extra_effects_on_replay: false`.

With a started runtime and tools already registered, the adapter interface is:

```python
from examples.react_agent_bridge import ReActToolBridge

bridge = ReActToolBridge(runtime)
observations = await bridge.execute_tool_calls(
    [{
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "fetch_document",
            "arguments": '{"document_id":"doc-1"}',
        },
    }],
    run_id="persisted-agent-run-1",
    step_idx=0,
)
# observations are role="tool" messages linked by tool_call_id;
# each JSON content contains job_id, terminal status, and result or error.
```

- Keys hash structured JSON `[run_id, step_idx, call_id]`; delimiter-containing IDs cannot collide through ambiguous concatenation. Persist and reuse the run/step/call identity, names, and arguments on replay.
- The complete batch is validated before admission, including historical key conflicts. Preflight scans stored history: this is an illustrative adapter, not an optimized serving API or transactional batch submission.
- Sequential is the default. A failed or unknown call blocks later calls with `NOT_DISPATCHED` observations. That is an adapter observation, not another durable job status.
- Parallel mode requires both `parallel=True, independent=True` and idempotent policies. Idempotency alone does not prove independence; the caller must assert it correctly.
- One `play_move` is permitted per batch. Turn boundaries, ordering across bridge instances/batches, and reconciliation before dependent work remain caller responsibilities.
- Register every tool before startup/bridge construction and do not mutate the registry afterward.
- Tool jobs are durable; reasoning, messages, external world state, and in-memory mock state are **not** checkpointed. Reopening the demo database can return cached results without rebuilding this process's mock move history.

## Scope and reproduction

```sh
uv run pytest -m acceptance -q
uv run pytest -m chaos -q
uv run pytest -q
PYTHONASYNCIODEBUG=1 uv run pytest -q
uv run --isolated --python 3.14 pytest -q
uvx ruff check .
uv build
```

The implementation deliberately uses database-pull scheduling plus a bounded unfinished backlog and fixed workers rather than a second in-memory job queue/WAL. This avoids reconciling two authoritative queues after crashes. Each transaction commits independently; custom group-commit batching is not implemented. Circuit breakers, per-tool token buckets, HTTP serving, arbitrary-code sandboxing, distributed coordination, retention/pruning, and full agent checkpointing are out of scope.

The project is standalone. The original ReAct repository is unchanged; the bridge demonstrates the tool-execution boundary rather than claiming a live integration.
