from __future__ import annotations

import asyncio
import json
import math
import pstats
from dataclasses import replace

import pytest

from benchmarks import bench_latency as bench


def test_percentiles_are_sorted_interpolated_and_include_endpoints():
    values = [30.0, 0.0, 20.0, 10.0]
    assert bench.percentile(values, 0) == 0
    assert bench.percentile(values, 0.5) == 15
    assert bench.percentile(values, 0.95) == pytest.approx(28.5)
    assert bench.percentile(values, 1) == 30
    assert bench.percentile([4.0], 0.99) == 4
    assert values == [30.0, 0.0, 20.0, 10.0]
    stats = bench.LatencyStats.from_values(values)
    assert stats.samples == 4 and stats.p50_ms == 15
    assert stats.p95_ms <= stats.p99_ms


@pytest.mark.parametrize(
    ("values", "quantile"),
    [
        ([], 0.5),
        ([1], -0.1),
        ([1], 1.1),
        ([1], math.nan),
        ([1], math.inf),
        ([-1], 0.5),
        ([math.nan], 0.5),
        ([math.inf], 0.5),
    ],
)
def test_invalid_percentiles_are_rejected(values, quantile):
    with pytest.raises(ValueError):
        bench.percentile(values, quantile)


@pytest.fixture
def sample():
    # Tool start may precede submit() acknowledgement: never subtract the ACK
    # from tool start or silently clamp a negative dispatch measurement.
    return bench.Sample(
        index=0,
        submitted_ns=1_000_000,
        admitted_ns=2_000_000,
        tool_started_ns=3_000_000,
        acknowledged_ns=4_000_000,
        tool_finished_ns=5_000_000,
        result_committed_ns=6_000_000,
        observed_ns=7_000_000,
    )


def test_intervals_use_commit_not_acknowledgement(sample):
    assert sample.intervals_ms() == {
        "admission_ack": 3,
        "admission_commit": 1,
        "queue_dispatch": 1,
        "completion_commit": 1,
        "end_to_end": 6,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submitted_ns", 0),
        ("admitted_ns", 8_000_000),
        ("tool_started_ns", 1_500_000),
        ("tool_finished_ns", 2_500_000),
        ("result_committed_ns", 4_500_000),
        ("observed_ns", 5_500_000),
        ("acknowledged_ns", 1_500_000),
        ("acknowledged_ns", 8_000_000),
    ],
)
def test_invalid_or_missing_timestamps_fail_instead_of_being_clamped(
    sample, field, value
):
    with pytest.raises(RuntimeError, match="timing"):
        replace(sample, **{field: value}).intervals_ms()


@pytest.mark.parametrize(
    "options",
    [
        {"jobs": 0},
        {"jobs": True},
        {"jobs": 1.5},
        {"warmup": -1},
        {"warmup": False},
        {"workers": 0},
        {"concurrency": 0},
        {"timeout": 0},
        {"timeout": math.inf},
        {"timeout": math.nan},
        {"timeout": True},
        {"timeout": "60"},
        {"synchronous_mode": "OFF"},
    ],
)
async def test_invalid_options_fail_before_runtime_start(options):
    with pytest.raises(ValueError):
        await bench.run_benchmark(**options)


@pytest.mark.parametrize(("mode", "warmup"), [("FULL", 3), ("NORMAL", 0)])
async def test_real_benchmark_validates_jobs_and_commit_hooks(mode, warmup):
    result = await bench.run_benchmark(
        mode,
        jobs=12,
        warmup=warmup,
        workers=2,
        concurrency=4,
        timeout=10,
    )
    assert result.synchronous_mode == mode
    assert result.jobs == result.successful_jobs == result.attempts == 12
    assert result.warmup_jobs == warmup
    assert result.workers == 2 and result.concurrency == 4
    assert result.elapsed_seconds > 0
    assert result.jobs_per_second == pytest.approx(12 / result.elapsed_seconds)
    assert result.asyncio_debug is asyncio.get_running_loop().get_debug()
    assert set(result.latency_ms) == {
        "admission_ack",
        "admission_commit",
        "queue_dispatch",
        "completion_commit",
        "end_to_end",
    }
    for stats in result.latency_ms.values():
        assert stats.samples == 12
        assert 0 <= stats.p50_ms <= stats.p95_ms <= stats.p99_ms
    assert json.loads(json.dumps(result.to_dict()))["successful_jobs"] == 12
    report = bench.format_report([result])
    assert f"| {mode} | 12 | 2 | 4 | {warmup} |" in report
    assert "DB commit to tool start" in report
    assert "not raw fsync time" in report
    assert "WARNING" not in report
    assert "WARNING: cProfile enabled" in bench.format_report([result], profiled=True)


def test_cli_writes_real_profile_json_and_environment(tmp_path, capsys):
    json_path = tmp_path / "nested" / "latency.json"
    profile_path = tmp_path / "nested" / "latency.prof"
    bench.main(
        [
            "--jobs",
            "4",
            "--warmup",
            "1",
            "--workers",
            "1",
            "--concurrency",
            "2",
            "--json",
            str(json_path),
            "--profile",
            str(profile_path),
        ]
    )
    report = json.loads(json_path.read_text())
    assert report["schema_version"] == 1 and report["profiled"] is True
    assert report["recorded_at_utc"]
    assert report["environment"]["python"]
    assert report["environment"]["sqlite"]
    assert report["environment"]["perf_counter_resolution_ns"] > 0
    assert [result["synchronous_mode"] for result in report["results"]] == [
        "FULL",
        "NORMAL",
    ]
    assert all(result["jobs"] == 4 for result in report["results"])
    assert pstats.Stats(str(profile_path)).get_stats_profile().func_profiles
    output = capsys.readouterr().out
    assert "WARNING: cProfile enabled" in output and "Environment:" in output


@pytest.mark.parametrize(
    "args",
    [
        ["--jobs", "0"],
        ["--warmup", "-1"],
        ["--workers", "0"],
        ["--concurrency", "0"],
        ["--timeout", "nan"],
        ["--timeout", "-1"],
        ["--synchronous", "OFF"],
        ["--json", "same", "--profile", "same"],
    ],
)
def test_cli_rejects_invalid_arguments_without_running(args):
    with pytest.raises(SystemExit) as exc:
        bench.main(args)
    assert exc.value.code == 2
