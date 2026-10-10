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
