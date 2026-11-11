"""Closed-loop no-op workload; timings are instrumentation-inclusive, not an SLA.

Run from the repository root: python -m benchmarks.bench_latency
Queue/dispatch starts at the database-thread post-admission-commit hook, NOT at
submit() return: a worker can start before the caller receives acknowledgement.
All timing uses perf_counter_ns; persisted wall-clock fields are not subtracted.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from runtime import (
    DurableToolRuntime,
    JobStatus,
    RuntimeConfig,
    SQLiteJobStore,
    ToolRegistry,
)
from runtime.models import JobRecord, ToolPolicy

Mode = Literal["FULL", "NORMAL"]


@dataclass(slots=True)
class Sample:
    index: int
    submitted_ns: int = 0
    admitted_ns: int = 0
    acknowledged_ns: int = 0
    tool_started_ns: int = 0
    tool_finished_ns: int = 0
    result_committed_ns: int = 0
    observed_ns: int = 0

    def intervals_ms(self) -> dict[str, float]:
        ordered = (
            self.submitted_ns,
            self.admitted_ns,
            self.tool_started_ns,
            self.tool_finished_ns,
            self.result_committed_ns,
            self.observed_ns,
        )
        if any(value <= 0 for value in ordered) or list(ordered) != sorted(ordered):
            raise RuntimeError(
                f"Missing or out-of-order timing for sample {self.index}"
            )
        if (
            self.acknowledged_ns < self.admitted_ns
            or self.acknowledged_ns > self.observed_ns
        ):
            raise RuntimeError(
                f"Invalid acknowledgement timing for sample {self.index}"
            )
        return {
            "admission_ack": (self.acknowledged_ns - self.submitted_ns) / 1e6,
            "admission_commit": (self.admitted_ns - self.submitted_ns) / 1e6,
            "queue_dispatch": (self.tool_started_ns - self.admitted_ns) / 1e6,
            "completion_commit": (self.result_committed_ns - self.tool_finished_ns)
            / 1e6,
            "end_to_end": (self.observed_ns - self.submitted_ns) / 1e6,
        }


class TimingStore(SQLiteJobStore):
    """Benchmark-only hooks around existing synchronous transactional operations.

    super() returns only AFTER COMMIT. Samples are inspected after clients finish;
    no concurrent iteration over the writer's dictionaries is required. These
    private-method overrides are intentionally tested against this store version.
    """

    def __init__(self, config: RuntimeConfig, samples: dict[int, Sample]) -> None:
        super().__init__(config)
        self.samples = samples
        self.by_job: dict[str, Sample] = {}

    def _admit_job(
        self,
        policy: ToolPolicy,
        args_json: str,
        fingerprint: str,
        key: str,
        now: float | None,
    ) -> tuple[JobRecord, bool]:
        job, created = super()._admit_job(policy, args_json, fingerprint, key, now)
        committed_ns = time.perf_counter_ns()
        if not created:
            raise RuntimeError("Benchmark unexpectedly reused an idempotency key")
        sample = self.samples[int(key)]
        sample.admitted_ns = committed_ns
        self.by_job[job.job_id] = sample
        return job, created

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
        job = super()._write_outcome(
            job_id, attempt, result_json, error_text, retryable, outcome_unknown, now
        )
        committed_ns = time.perf_counter_ns()
        if job.status is not JobStatus.SUCCEEDED or attempt != 1:
            raise RuntimeError("No-op benchmark encountered a failure or retry")
        self.by_job[job_id].result_committed_ns = committed_ns
        return job


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear interpolation at (n - 1) * quantile; no nearest-rank ambiguity."""
    if not values:
        raise ValueError("Cannot take a percentile of no values")
    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency values must be finite and nonnegative")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> LatencyStats:
        return cls(len(values), *(percentile(values, q) for q in (0.5, 0.95, 0.99)))


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    synchronous_mode: Mode
    jobs: int
    warmup_jobs: int
    workers: int
    concurrency: int
    elapsed_seconds: float
    jobs_per_second: float
    successful_jobs: int
    attempts: int
    asyncio_debug: bool
    latency_ms: dict[str, LatencyStats]

    def to_dict(self) -> dict:
        return asdict(self)


def _integer(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


async def run_benchmark(
    synchronous_mode: Mode = "FULL",
    *,
    jobs: int = 1000,
    warmup: int = 100,
    workers: int = 16,
    concurrency: int = 16,
    timeout: float = 60,
) -> BenchmarkResult:
    for value, name in (
        (jobs, "jobs"),
        (workers, "workers"),
        (concurrency, "concurrency"),
    ):
        _integer(value, name)
    _integer(warmup, "warmup", minimum=0)
    if synchronous_mode not in ("FULL", "NORMAL"):
        raise ValueError("synchronous_mode must be FULL or NORMAL")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    samples = {index: Sample(index) for index in range(-warmup, jobs)}
    registry = ToolRegistry()

    @registry.tool(idempotent=True, timeout_ms=5000, max_retries=0)
    async def noop(x: int) -> int:
        sample = samples[x]
        sample.tool_started_ns = time.perf_counter_ns()
        sample.tool_finished_ns = time.perf_counter_ns()
        return x

    async def clients(runtime: DurableToolRuntime, indices: range) -> None:
        remaining = iter(indices)

        async def client() -> None:
            for index in remaining:
                sample = samples[index]
                sample.submitted_ns = time.perf_counter_ns()
                job_id = await runtime.submit("noop", {"x": index}, str(index))
                sample.acknowledged_ns = time.perf_counter_ns()
                job = await runtime.wait_for_terminal(job_id)
                sample.observed_ns = time.perf_counter_ns()
                if (
                    job.status is not JobStatus.SUCCEEDED
                    or job.attempt_count != 1
                    or job.result != index
                ):
                    raise RuntimeError(
                        f"No-op job {index} did not complete correctly once"
                    )

        tasks = [
            asyncio.create_task(client()) for _ in range(min(concurrency, len(indices)))
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    with TemporaryDirectory(prefix="durable-runtime-bench-") as directory:
        config = RuntimeConfig(
            db_path=str(Path(directory) / "jobs.sqlite3"),
            max_workers=workers,
            max_outstanding_jobs=concurrency,
            lease_duration_ms=5000,
            heartbeat_interval_ms=1000,
            poll_interval_ms=5,
            recovery_interval_ms=100,
            synchronous_mode=synchronous_mode,
        )
        runtime = DurableToolRuntime(config, registry)
        runtime.store = TimingStore(config, samples)
        async with asyncio.timeout(timeout):
            async with runtime:
                await clients(runtime, range(-warmup, 0))
                started_ns = time.perf_counter_ns()
                await clients(runtime, range(jobs))
                elapsed_ns = time.perf_counter_ns() - started_ns
    intervals = [samples[index].intervals_ms() for index in range(jobs)]
    metrics = {
        name: LatencyStats.from_values([sample[name] for sample in intervals])
        for name in intervals[0]
    }
    elapsed_seconds = elapsed_ns / 1e9
    return BenchmarkResult(
        synchronous_mode,
        jobs,
        warmup,
        workers,
        concurrency,
        elapsed_seconds,
        jobs / elapsed_seconds,
        jobs,
        jobs,
        asyncio.get_running_loop().get_debug(),
        metrics,
    )


def environment() -> dict:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin" and (sysctl := shutil.which("sysctl")):
        try:
            cpu = subprocess.run(
                [sysctl, "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "sqlite": sqlite3.sqlite_version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "perf_counter_resolution_ns": time.get_clock_info("perf_counter").resolution
        * 1e9,
    }


def format_report(results: Sequence[BenchmarkResult], *, profiled: bool = False) -> str:
    lines = [
        "# Local no-op runtime benchmark",
        "",
        "Closed-loop clients; timings include instrumentation and result-observer queries.",
        "Fresh temporary database per mode; warmup excluded. No network or LLM latency.",
    ]
    if profiled:
        lines.append(
            "WARNING: cProfile enabled; do not compare these timings with an unprofiled baseline."
        )
    lines.extend(
        [
            "",
            "| Mode | Jobs | Workers | Clients | Warmup | Elapsed (s) | Jobs/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.synchronous_mode} | {result.jobs} | {result.workers} | "
            f"{result.concurrency} | {result.warmup_jobs} | {result.elapsed_seconds:.3f} | "
            f"{result.jobs_per_second:.1f} |"
        )
    names = {
        "admission_ack": "Admission: submit to acknowledgement",
        "admission_commit": "Admission: submit to DB commit",
        "queue_dispatch": "Queue + dispatch: DB commit to tool start",
        "completion_commit": "Completion: tool return to DB commit",
        "end_to_end": "End-to-end: submit to terminal observation",
    }
    lines.extend(
        [
            "",
            "| Mode | Metric | p50 (ms) | p95 (ms) | p99 (ms) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        for metric, stats in result.latency_ms.items():
            lines.append(
                f"| {result.synchronous_mode} | {names[metric]} | "
                f"{stats.p50_ms:.3f} | {stats.p95_ms:.3f} | {stats.p99_ms:.3f} |"
            )
    lines.extend(
        [
            "",
            "All samples verified: SUCCEEDED, one attempt, correct result, ordered timestamps.",
            "Percentiles: linear interpolation at (n - 1) * q. Throughput includes client observation.",
            "Commit timestamps are sampled immediately after transactional store methods return.",
            "These intervals are not raw fsync time; scheduling, serialization and writer queueing are included.",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--synchronous", choices=("FULL", "NORMAL", "BOTH"), default="BOTH"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write aggregate measurements and environment metadata",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Write cProfile stats; profiled timings are not a baseline",
    )
    args = parser.parse_args(argv)
    for value, name, minimum in (
        (args.jobs, "jobs", 1),
        (args.workers, "workers", 1),
        (args.concurrency, "concurrency", 1),
        (args.warmup, "warmup", 0),
    ):
        try:
            _integer(value, name, minimum=minimum)
        except ValueError as exc:
            parser.error(str(exc))
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.json and args.profile and args.json.resolve() == args.profile.resolve():
        parser.error("--json and --profile must refer to different files")

    async def run() -> list[BenchmarkResult]:
        modes = (
            ("FULL", "NORMAL") if args.synchronous == "BOTH" else (args.synchronous,)
        )
        return [
            await run_benchmark(
                mode,
                jobs=args.jobs,
                warmup=args.warmup,
                workers=args.workers,
                concurrency=args.concurrency,
                timeout=args.timeout,
            )
            for mode in modes
        ]

    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()
    try:
        results = asyncio.run(run())
    finally:
        if profiler is not None:
            profiler.disable()
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.profile))
    report = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment(),
        "profiled": profiler is not None,
        "results": [result.to_dict() for result in results],
    }
    print(format_report(results, profiled=profiler is not None))
    print("\nEnvironment: " + json.dumps(report["environment"], sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
