"""Load optional project-local environment configuration safely.

JARVIS remains usable without a .env file or cloud API keys. Real environment
variables take precedence over values in the local .env file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_project_env() -> None:
    """Load the project-root .env without overriding existing process vars."""
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or not (key.replace("_", "a").isalnum()) or key[0].isdigit():
            continue
        if key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ[key] = value

    # Groq is OpenAI-compatible. Let users configure just GROQ_API_KEY in .env;
    # legacy JARVIS settings remain supported and take precedence when provided.
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key and not os.environ.get("JARVIS_MODEL_API_KEY", "").strip():
        os.environ["JARVIS_MODEL_API_KEY"] = groq_key
    if groq_key and not os.environ.get("JARVIS_MODEL_BASE_URL", "").strip():
        os.environ["JARVIS_MODEL_BASE_URL"] = "https://api.groq.com/openai/v1"
    if groq_key and not os.environ.get("JARVIS_MODEL_NAME", "").strip():
        os.environ["JARVIS_MODEL_NAME"] = "openai/gpt-oss-120b"


load_project_env()
