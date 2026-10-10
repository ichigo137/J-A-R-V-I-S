from __future__ import annotations

from jarvis.cli import _select_local_model


GiB = 1024**3


def _model(key: str, size: int, *, kind: str = "llm") -> dict[str, object]:
    return {"type": kind, "modelKey": key, "sizeBytes": size}


def test_selects_smallest_model_that_fits_conservative_runtime_budget():
    models = [
        _model("large-7b", 4 * GiB),
        _model("small-1b", GiB),
        _model("medium-3b", 2 * GiB),
        _model("embedding", GiB, kind="embedding"),
    ]

    selected = _select_local_model(models, available_memory_bytes=8 * GiB)

    assert selected is not None
    assert selected["modelKey"] == "small-1b"


def test_returns_none_when_no_model_fits_available_memory():
    models = [_model("4b", 3 * GiB)]

    assert _select_local_model(models, available_memory_bytes=2 * GiB) is None


def test_requested_model_is_still_checked_against_memory_budget():
    models = [_model("tiny", GiB), _model("large", 4 * GiB)]

    selected = _select_local_model(
        models,
        available_memory_bytes=8 * GiB,
        requested_model_key="large",
    )

    assert selected is None


def test_requested_model_wins_when_it_fits():
    models = [_model("tiny", GiB), _model("medium", 2 * GiB)]

    selected = _select_local_model(
        models,
        available_memory_bytes=8 * GiB,
        requested_model_key="medium",
    )

    assert selected is not None
    assert selected["modelKey"] == "medium"


def test_ignores_invalid_model_metadata():
    models = [
        {"type": "llm", "modelKey": "missing-size"},
        _model("", GiB),
        _model("not-a-language-model", GiB, kind="embedding"),
        _model("valid", GiB),
    ]

    selected = _select_local_model(models, available_memory_bytes=8 * GiB)

    assert selected is not None
    assert selected["modelKey"] == "valid"
