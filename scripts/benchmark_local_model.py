from __future__ import annotations

"""Measure local OpenAI-compatible inference without silently stressing low-RAM hosts.

Run from the repository root with:
    .venv\\Scripts\\python.exe scripts\\benchmark_local_model.py

The benchmark sends a single short request after a configurable free-RAM guard.
It does not load or unload models. LM Studio must already be running with a model
loaded. Results include request latency, token throughput when reported, and
system/process memory snapshots; they are diagnostic measurements, not guarantees.
"""

import argparse
import json
import os
import sys
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


def _model_process_memory() -> dict[str, int]:
    """Return best-effort RSS totals for visible local model-runtime processes."""
    patterns = ("lm studio", "llama-server", "llama.cpp", "llmster", "ollama")
    matched: list[psutil.Process] = []
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if any(pattern in name for pattern in patterns):
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
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Model endpoint returned a non-object JSON response")
    return decoded


def run_benchmark(
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    timeout: float = 60.0,
    min_available_mib: int = DEFAULT_MIN_AVAILABLE_MIB,
    max_tokens: int = 32,
) -> dict[str, Any]:
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

    memory_at_start = psutil.virtual_memory()
    process_at_start = _model_process_memory()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = _post_json(f"{root}/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - started
    memory_after = psutil.virtual_memory()
    process_after = _model_process_memory()

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    completion_tokens = usage.get("completion_tokens")
    content = ""
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            content = message["content"]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": root,
        "model": model_id,
        "latency_seconds": round(elapsed, 3),
        "completion_tokens_reported": completion_tokens,
        "tokens_per_second_reported": (
            round(completion_tokens / elapsed, 2)
            if isinstance(completion_tokens, int) and elapsed > 0
            else None
        ),
        "response_text": content,
        "system_memory": {
            "total_bytes": memory_before.total,
            "available_before_bytes": memory_before.available,
            "available_at_request_start_bytes": memory_at_start.available,
            "available_after_bytes": memory_after.available,
            "available_delta_bytes": memory_after.available - memory_before.available,
        },
        "model_runtime_processes": {
            "before": process_at_start,
            "after": process_after,
            "rss_delta_bytes": process_after["rss_bytes"] - process_at_start["rss_bytes"],
            "note": "Best-effort process-name match; model memory may be in child processes not matched here.",
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
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report")
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            timeout=args.timeout,
            min_available_mib=args.min_available_mib,
            max_tokens=args.max_tokens,
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
