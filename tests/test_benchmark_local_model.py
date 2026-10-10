from __future__ import annotations

import pytest

from scripts import benchmark_local_model as benchmark


def test_benchmark_stops_before_network_when_available_ram_is_below_guard(monkeypatch):
    low_memory = type(
        "Memory",
        (),
        {"available": 256 * 1024 * 1024, "total": 8 * 1024**3},
    )()
    monkeypatch.setattr(benchmark.psutil, "virtual_memory", lambda: low_memory)

    with pytest.raises(RuntimeError, match="Safety stop"):
        benchmark.run_benchmark(min_available_mib=1024)


def test_process_memory_snapshot_has_explicit_units():
    snapshot = benchmark._model_process_memory()

    assert set(snapshot) == {"process_count", "rss_bytes"}
    assert snapshot["process_count"] >= 0
    assert snapshot["rss_bytes"] >= 0


@pytest.mark.parametrize(
    ("values", "percentile", "expected"),
    [
        ([1.0], 0.95, 1.0),
        ([1.0, 2.0, 3.0], 0.5, 2.0),
        ([1.0, 2.0, 3.0, 4.0], 0.95, 3.85),
        ([], 0.95, None),
    ],
)
def test_percentile_interpolates_and_handles_empty(values, percentile, expected):
    actual = benchmark._percentile(values, percentile)
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_requests": -1}, "warmup_requests"),
        ({"requests": 0}, "requests"),
        ({"max_tokens": 0}, "max_tokens"),
    ],
)
def test_invalid_benchmark_counts_fail_before_network(kwargs, message):
    with pytest.raises(ValueError, match=message):
        benchmark.run_benchmark(**kwargs)
