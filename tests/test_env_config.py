from __future__ import annotations

import os
from pathlib import Path

from jarvis.env_config import load_project_env


def test_load_project_env_preserves_existing_environment(monkeypatch, tmp_path: Path):
    project = tmp_path / "src" / "jarvis"
    project.mkdir(parents=True)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nFROM_FILE=hello\nEXISTING=from-file\nQUOTED=\"two words\"\nINLINE=value # note\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("jarvis.env_config.__file__", str(project / "env_config.py"))
    monkeypatch.delenv("FROM_FILE", raising=False)
    monkeypatch.setenv("EXISTING", "from-process")
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("INLINE", raising=False)

    load_project_env()

    assert os.environ["FROM_FILE"] == "hello"
    assert os.environ["EXISTING"] == "from-process"
    assert os.environ["QUOTED"] == "two words"
    assert os.environ["INLINE"] == "value"


def test_missing_env_file_is_ok(monkeypatch, tmp_path: Path):
    project = tmp_path / "src" / "jarvis"
    project.mkdir(parents=True)
    monkeypatch.setattr("jarvis.env_config.__file__", str(project / "env_config.py"))
    load_project_env()  # should not raise if .env doesn't exist


def test_groq_key_configures_openai_compatible_pipeline(monkeypatch, tmp_path: Path):
    project = tmp_path / "src" / "jarvis"
    project.mkdir(parents=True)
    (tmp_path / ".env").write_text("GROQ_API_KEY=test-groq-key\n", encoding="utf-8")
    monkeypatch.setattr("jarvis.env_config.__file__", str(project / "env_config.py"))
    for name in ("GROQ_API_KEY", "JARVIS_MODEL_API_KEY", "JARVIS_MODEL_BASE_URL", "JARVIS_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)

    load_project_env()

    assert os.environ["JARVIS_MODEL_API_KEY"] == "test-groq-key"
    assert os.environ["JARVIS_MODEL_BASE_URL"] == "https://api.groq.com/openai/v1"
    assert os.environ["JARVIS_MODEL_NAME"] == "openai/gpt-oss-120b"


def test_explicit_jarvis_provider_settings_take_precedence(monkeypatch, tmp_path: Path):
    project = tmp_path / "src" / "jarvis"
    project.mkdir(parents=True)
    (tmp_path / ".env").write_text("GROQ_API_KEY=test-groq-key\n", encoding="utf-8")
    monkeypatch.setattr("jarvis.env_config.__file__", str(project / "env_config.py"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("JARVIS_MODEL_API_KEY", "explicit-key")
    monkeypatch.setenv("JARVIS_MODEL_BASE_URL", "https://example.invalid/v1")

    load_project_env()

    assert os.environ["JARVIS_MODEL_API_KEY"] == "explicit-key"
    assert os.environ["JARVIS_MODEL_BASE_URL"] == "https://example.invalid/v1"
