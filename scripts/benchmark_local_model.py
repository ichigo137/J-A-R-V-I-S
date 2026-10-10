from __future__ import annotations

"""Safely benchmark an already-loaded local OpenAI-compatible model.

Does not load or unload models. Defaults to a 1 GiB free-RAM guard, one warm-up,
and five measured requests. Streaming mode estimates time-to-first-chunk and
generation rate; token counts depend on what the server reports.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_PROMPT = "Reply with exactly: JARVIS_BENCHMARK_OK"
DEFAULT_MIN_AVAILABLE_MIB = 1024
_MEMORY_SAMPLE_INTERVAL = 0.1
_PROCESS_PATTERNS = ("lm studio", "llama-server", "llama.cpp", "llmster", "ollama")


def _model_process_memory() -> dict[str, int]:
    matched: list[psutil.Process] = []
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if any(pattern in name for pattern in _PROCESS_PATTERNS):
                matched.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "process_count": len(matched),
        "rss_bytes": sum(
            proc.info["memory_info"].rss
            for proc in matched
            if proc.info.get("memory_info") is not None
        ),
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Model endpoint returned a non-object JSON response")
    return decoded


def _stream_request(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[float, float | None, int | None, str]:
    """Return elapsed, TTFC, reported completion tokens, and response text."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    first_chunk: float | None = None
    chunks = 0
    pieces: list[str] = []
    reported_tokens: int | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if first_chunk is None:
                first_chunk = time.perf_counter() - started
            usage = event.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                reported_tokens = usage["completion_tokens"]
            choices = event.get("choices")
            if isinstance(choices, list) and choices:
                delta = choices[0].get("delta", {})
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    piece = delta["content"]
                    if piece:
                        chunks += 1
                        pieces.append(piece)
    elapsed = time.perf_counter() - started
    # If server omits usage, chunks are not tokens. Do not mislabel them as tokens.
    return elapsed, first_chunk, reported_tokens, "".join(pieces)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def run_benchmark(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    timeout: float = 60.0,
    min_available_mib: int = DEFAULT_MIN_AVAILABLE_MIB,
    max_tokens: int = 32,
    warmup_requests: int = 1,
    requests: int = 5,
    stream: bool = False,
) -> dict[str, Any]:
    if warmup_requests < 0:
        raise ValueError("warmup_requests must be >= 0")
    if requests < 1:
        raise ValueError("requests must be >= 1")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    memory_before = psutil.virtual_memory()
    available_mib = memory_before.available / (1024 * 1024)
    if available_mib < min_available_mib:
        raise RuntimeError(
            f"Safety stop: only {available_mib:.0f} MiB RAM available; "
            f"benchmark requires at least {min_available_mib} MiB. "
            "Free memory or pass --min-available-mib 0 only if you deliberately "
            "accept the risk of stressing this machine."
        )

    root = base_url.rstrip("/")
    model_id = model
    if not model_id:
        request = urllib.request.Request(f"{root}/models", method="GET")
        with urllib.request.urlopen(request, timeout=min(timeout, 5.0)) as response:
            models_payload = json.loads(response.read().decode("utf-8"))
        models = models_payload.get("data", []) if isinstance(models_payload, dict) else []
        if not models:
            raise RuntimeError(f"No models returned by {root}/models")
        model_id = str(models[0].get("id", ""))
        if not model_id:
            raise RuntimeError("The endpoint's first model has no ID; pass --model explicitly")

    stop_sampler = threading.Event()
    memory_samples: list[int] = [memory_before.available]

    def sample_memory() -> None:
        while not stop_sampler.wait(_MEMORY_SAMPLE_INTERVAL):
            try:
                memory_samples.append(psutil.virtual_memory().available)
            except (OSError, RuntimeError):
                pass

    sampler = threading.Thread(target=sample_memory, name="jarvis-memory-sampler", daemon=True)
    process_before = _model_process_memory()
    sampler.start()
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    measurements: list[dict[str, Any]] = []
    warmups_completed = 0
    error: str | None = None
    try:
        for index in range(warmup_requests + requests):
            try:
                if stream:
                    elapsed, ttfc, tokens, content = _stream_request(
                        f"{root}/chat/completions", payload, timeout
                    )
                else:
                    started = time.perf_counter()
                    response = _post_json(f"{root}/chat/completions", payload, timeout)
                    elapsed = time.perf_counter() - started
                    ttfc = None
                    usage = response.get("usage")
                    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                    content = ""
                    choices = response.get("choices")
                    if isinstance(choices, list) and choices:
                        message = choices[0].get("message", {})
                        if isinstance(message, dict) and isinstance(message.get("content"), str):
                            content = message["content"]
            except (OSError, ValueError, urllib.error.URLError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break

            if index < warmup_requests:
                warmups_completed += 1
            else:
                measurements.append({
                    "latency_seconds": elapsed,
                    "time_to_first_chunk_seconds": ttfc,
                    "completion_tokens_reported": tokens,
                    "response_text": content,
                    "reported_tokens_per_second_end_to_end": (
                        round(tokens / elapsed, 2)
                        if isinstance(tokens, int) and elapsed > 0 else None
                    ),
                    "reported_generation_tokens_per_second": (
                        round(tokens / (elapsed - ttfc), 2)
                        if isinstance(tokens, int) and ttfc is not None and elapsed > ttfc
                        else None
                    ),
                })
    finally:
        stop_sampler.set()
        sampler.join(timeout=1.0)
    memory_after = psutil.virtual_memory()
    process_after = _model_process_memory()
    latencies = [float(item["latency_seconds"]) for item in measurements]
    ttfc_values = [
        float(item["time_to_first_chunk_seconds"])
        for item in measurements if item["time_to_first_chunk_seconds"] is not None
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": root,
        "model": model_id,
        "mode": "streaming" if stream else "non_streaming",
        "warmup_requests_requested": warmup_requests,
        "warmup_requests_completed": warmups_completed,
        "measured_requests_requested": requests,
        "measured_requests_completed": len(measurements),
        "error": error,
        "latency_seconds": {
            "samples": [round(value, 4) for value in latencies],
            "median": round(statistics.median(latencies), 4) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 4) if latencies else None,
        },
        "time_to_first_chunk_seconds": {
            "samples": [round(value, 4) for value in ttfc_values],
            "median": round(statistics.median(ttfc_values), 4) if ttfc_values else None,
            "p95": round(_percentile(ttfc_values, 0.95), 4) if ttfc_values else None,
            "note": "Streaming event arrival, not guaranteed to equal the first generated token.",
        },
        "requests": measurements,
        "system_memory": {
            "total_bytes": memory_before.total,
            "available_before_bytes": memory_before.available,
            "available_after_bytes": memory_after.available,
            "available_delta_bytes": memory_after.available - memory_before.available,
            "minimum_available_sampled_bytes": min(memory_samples),
            "peak_available_ram_drop_bytes": max(0, memory_before.available - min(memory_samples)),
            "sampling_interval_seconds": _MEMORY_SAMPLE_INTERVAL,
        },
        "model_runtime_processes": {
            "before": process_before,
            "after": process_after,
            "rss_delta_bytes": process_after["rss_bytes"] - process_before["rss_bytes"],
            "note": "Best-effort process-name match; RSS and 100ms RAM samples may miss true peak usage.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("JARVIS_LOCAL_MODEL_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("JARVIS_LOCAL_MODEL_NAME") or None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-available-mib", type=int, default=DEFAULT_MIN_AVAILABLE_MIB)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--stream", action="store_true", help="Measure streaming event arrival and TTFC")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report")
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            base_url=args.base_url, model=args.model, prompt=args.prompt,
            timeout=args.timeout, min_available_mib=args.min_available_mib,
            max_tokens=args.max_tokens, warmup_requests=args.warmup_requests,
            requests=args.requests, stream=args.stream,
        )
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"Benchmark not completed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
