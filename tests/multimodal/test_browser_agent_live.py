"""Opt-in real-model/browser smoke test. Skipped unless explicitly enabled."""

import os
import socket
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("JARVIS_RUN_LIVE_BROWSER_TESTS") != "1",
    reason="set JARVIS_RUN_LIVE_BROWSER_TESTS=1 to run live browser tests",
)


def test_live_search_then_inspect():
    try:
        with socket.create_connection(("127.0.0.1", 1234), timeout=1.0):
            pass
    except OSError:
        pytest.skip("LM Studio is not listening on 127.0.0.1:1234")

    from jarvis.multimodal.jarvis_voice import JarvisVoiceAssistant

    assistant = JarvisVoiceAssistant(
        llm_provider="auto",
        llm_model=os.environ.get("JARVIS_LOCAL_MODEL", "qwen3-1.7b"),
        greeting=False,
        verbose=True,
    )
    # Resolve lazily before checking: LLMEngine initializes providers on first use.
    if not assistant.llm.is_online:
        pytest.skip(
            "JARVIS could not initialize the local OpenAI-compatible client: "
            + "; ".join(assistant.llm.resolve_errors)
        )

    # Never let a live smoke test wait the engine's normal 120-second timeout.
    assistant.llm._client = assistant.llm._client.with_options(timeout=20.0, max_retries=0)
    proposals = []
    original = assistant.llm.plan_tool

    def record(**kwargs):
        result = original(**kwargs)
        if result:
            proposals.append(result.get("name"))
        return result

    assistant.llm.plan_tool = record
    try:
        first = assistant._think("Search the web for Python 3.12 release notes")
        assert proposals and proposals[0] == "browser_search", (proposals, first)
        assert "failed" not in first.lower(), first

        second = assistant._think("Describe the page currently open in the browser")
        assert len(proposals) >= 2 and proposals[1] == "browser_inspect", (proposals, second)
        assert "failed" not in second.lower(), second
        assert len(second.strip()) > 20, second
    finally:
        if assistant._browser_agent is not None:
            assistant._browser_agent.close()
