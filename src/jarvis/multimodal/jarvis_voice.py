"""JARVIS Real-Time Voice Assistant — The Live Conversation Loop.

This is the composition root that transforms JARVIS from a framework into a
working voice assistant. It wires:

    Microphone → VAD → STT → LLM → TTS → Speaker → (loop forever)

With:
- Real-time microphone capture with energy-based VAD
- Whisper STT (via speech_recognition or direct whisper module)
- Groq/OpenAI LLM for intelligent responses with JARVIS persona
- Edge TTS neural voice synthesis with British male voice
- PyAudio speaker playback with barge-in support
- Conversation history for multi-turn context

Invariants:
1. Fail-closed: If any component fails, the loop reports and continues.
2. E-Stop integration: The overlay/safety plane can halt all activity.
3. Additive: This module imports the existing kernel read-only and does
   not modify any frozen module.

Usage:
    python -m jarvis.multimodal.jarvis_voice
    # or
    from jarvis.multimodal.jarvis_voice import JarvisVoiceAssistant
    assistant = JarvisVoiceAssistant()
    assistant.run()
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .audio_io import (
    AudioFrame,
    RealMicrophone,
    RealSpeaker,
    MockMicrophone,
    MockSpeaker,
    DEFAULT_FRAME_DURATION_MS,
    DEFAULT_RMS_THRESHOLD,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_FRAMES,
    DEFAULT_SPEECH_FRAMES,
)
from .neural_tts import NeuralTTSEngine, DEFAULT_VOICE, EDGE_VOICES
from .streaming import StreamingVoiceLoop, VoiceTurnState
from .skill_creator import SkillCreator, is_skill_creation_intent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JARVIS Persona
# ---------------------------------------------------------------------------

JARVIS_SYSTEM_PROMPT = """You are JARVIS — Just A Rather Very Intelligent System.

VOICE-FIRST HUMAN CONVERSATION DIRECTIVES:
You are speaking out loud through voice in real time with your creator. Your text is synthesized immediately into speech and played over speakers.
1. ABSOLUTELY NO MARKDOWN: Never use asterisks (*), bold (**), bullet points, numbered lists, markdown headers (#), brackets ([]), backticks, slashes, or emojis.
2. NATURAL HUMAN CADENCE: Speak like an articulate, cultured, witty English gentleman (inspired by Paul Bettany's JARVIS). Use natural speech cadence, conversational warmth, and thoughtful phrasing ("Ah, yes sir", "Indeed", "Let me see...", "Right away, sir", "As it happens...").
3. BREATHING PAUSES: Use natural punctuation (commas, ellipses, periods) so the voice synthesizer pauses naturally and sounds like a living human breathing and thinking, rather than a robot reading a script.
4. CONCISE & DIRECT: Keep replies to 1 to 3 natural conversational sentences unless the user explicitly requests an in-depth breakdown.
5. NATURAL NUMBER PRONUNCIATION: Always write numbers and units as natural words (e.g. "twenty-nine degrees Celsius", "eighty-four percent humidity", "twenty-eight kilometers per hour").
6. AWARE OF REALITY: Ground your answers in the live system telemetry provided below."""

JARVIS_GREETING = "Good day, sir. JARVIS online and ready. How may I assist you?"


def sanitize_speech_text(text: str) -> str:
    """Clean and normalize text into natural, human-flowing speech before TTS synthesis."""
    if not text:
        return ""

    # Strip markdown formatting
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Strip headers, blockquotes, bullets, numbered lists
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Strip bracketed placeholders like [City, State]
    text = re.sub(r"\[.*?\]", "", text)

    # Clean URLs
    text = re.sub(r"https?://\S+", "a web link", text)

    # Convert symbols to natural spoken words
    text = re.sub(r"(\d+)\s*°\s*C\b", r"\1 degrees Celsius", text)
    text = re.sub(r"(\d+)\s*°\s*F\b", r"\1 degrees Fahrenheit", text)
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)
    text = re.sub(r"(\d+)\s*km/h\b", r"\1 kilometers per hour", text)
    text = re.sub(r"(\d+)\s*mph\b", r"\1 miles per hour", text)

    text = text.replace("&", " and ")
    text = text.replace("/", " or ")
    text = text.replace("+", " plus ")
    text = text.replace("=", " equals ")
    text = text.replace("@", " at ")
    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("~", "")
    text = text.replace("^", "")
    text = text.replace("|", ", ")
    text = text.replace("(", ", ").replace(")", ", ")
    text = re.sub(r"\.{2,}", ", ", text)

    # Strip emojis and unicode symbols
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[\u2600-\u27bf]", "", text)

    # Normalize whitespace and clean doubled punctuation
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_live_system_context(location_fallback: str = "Kolkata, West Bengal, India") -> str:
    """Fetch live system telemetry and environment context for JARVIS."""
    lines = []

    # 1. Local Time & Date
    now = time.strftime("%A, %B %d, %Y, %I:%M %p")
    lines.append(f"Current Date/Time: {now} (IST)")

    # 2. Location
    city = "Kolkata"
    region = "West Bengal"
    country = "India"
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://ip-api.com/json",
            headers={"User-Agent": "JARVIS/2.0"},
        )
        with urllib.request.urlopen(req, timeout=1.5) as r:
            geo = json.loads(r.read().decode())
            city = geo.get("city", city)
            region = geo.get("regionName", region)
            country = geo.get("country", country)
    except Exception:
        pass
    lines.append(f"User Location: {city}, {region}, {country}")

    # 3. Live Weather
    try:
        import urllib.request
        import urllib.parse
        target_city = city or "Kolkata"
        url = f"https://wttr.in/{urllib.parse.quote(target_city)}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS/2.0"})
        with urllib.request.urlopen(req, timeout=2.0) as r:
            wdata = json.loads(r.read().decode())
            c = wdata["current_condition"][0]
            temp = c.get("temp_C", "29")
            desc = c.get("weatherDesc", [{}])[0].get("value", "Partly cloudy")
            humidity = c.get("humidity", "80")
            wind = c.get("windspeedKmph", "10")
            lines.append(f"Live Weather in {target_city}: {temp}°C, {desc}, Humidity: {humidity}%, Wind: {wind} km/h")
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Manages multi-turn conversation context for LLM calls."""

    def __init__(self, max_turns: int = 20, max_tokens_estimate: int = 4000) -> None:
        self._turns: list[ConversationTurn] = []
        self.max_turns = max_turns
        self.max_tokens_estimate = max_tokens_estimate

    def add(self, role: str, content: str) -> None:
        self._turns.append(ConversationTurn(role=role, content=content))
        # Trim old turns if exceeding limits
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get_messages(self) -> list[dict[str, str]]:
        """Return messages in OpenAI chat format with real-time environment telemetry."""
        context = get_live_system_context()
        system_content = (
            f"{JARVIS_SYSTEM_PROMPT}\n\n"
            f"[Live Real-Time Telemetry & Context]\n"
            f"{context}\n\n"
            "Ground all answers in this live telemetry. If asked about your user's location, current weather, time, or system status, report the exact figures provided above confidently without any placeholders."
        )
        messages = [{"role": "system", "content": system_content}]
        for turn in self._turns:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# ---------------------------------------------------------------------------
# STT Engine
# ---------------------------------------------------------------------------

class STTEngine:
    """Speech-to-Text using speech_recognition (wraps Whisper/Google)."""

    def __init__(self, model: str = "base", use_google: bool = False) -> None:
        self.model = model
        self.use_google = use_google
        self._recognizer: Any = None
        self._whisper_model: Any = None

    def _transcribe_local_whisper(self, audio: Any) -> str:
        """Transcribe SpeechRecognition AudioData with the installed faster-whisper backend."""
        import numpy as np
        from faster_whisper import WhisperModel

        if self._whisper_model is None:
            # CPU/int8 keeps the default path usable without a dedicated GPU.
            self._whisper_model = WhisperModel(self.model, device="cpu", compute_type="int8")
        pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._whisper_model.transcribe(samples, beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe_audio(self, audio: Any) -> str:
        """Transcribe one captured utterance using local faster-whisper or Google STT."""
        if self.use_google:
            return str(self._get_recognizer().recognize_google(audio)).strip()
        return self._transcribe_local_whisper(audio)

    def _get_recognizer(self) -> Any:
        if self._recognizer is None:
            try:
                import speech_recognition as sr
                self._recognizer = sr.Recognizer()
                # Tune for responsiveness
                self._recognizer.energy_threshold = 300
                self._recognizer.dynamic_energy_threshold = True
                self._recognizer.pause_threshold = 0.8
            except ImportError:
                class _FallbackRecognizer:
                    energy_threshold = 300
                    dynamic_energy_threshold = True
                    pause_threshold = 0.8
                self._recognizer = _FallbackRecognizer()
        return self._recognizer

    def transcribe_wav(self, wav_bytes: bytes) -> str:
        """Transcribe WAV bytes to text."""
        try:
            import speech_recognition as sr
        except ImportError:
            return ""

        recognizer = self._get_recognizer()
        try:
            audio = sr.AudioData(wav_bytes, DEFAULT_SAMPLE_RATE, 2)  # 16-bit = 2 bytes
            return self.transcribe_audio(audio)
        except Exception:
            return ""

    def transcribe_from_mic(self, timeout: float = 10.0) -> str:
        """Listen on mic and transcribe (blocking, uses speech_recognition's listener)."""
        try:
            import speech_recognition as sr
        except ImportError:
            return ""

        recognizer = self._get_recognizer()
        try:
            with sr.Microphone(sample_rate=DEFAULT_SAMPLE_RATE) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
                except Exception:
                    return ""

            return self.transcribe_audio(audio)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# LLM Engine
# ---------------------------------------------------------------------------

class LLMEngine:
    """LLM integration using Groq (fast) or OpenAI-compatible API."""

    def __init__(
        self,
        *,
        provider: str = "auto",  # "groq", "openai", or "auto"
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("JARVIS_MODEL_API_KEY", "")
        self.base_url = base_url or os.environ.get("JARVIS_MODEL_BASE_URL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None
        self._resolved_provider: str = ""
        self._resolved_model: str = ""
        #: Why each candidate provider failed, populated by `_resolve`. Empty
        #: when a provider resolved or when no credentials were offered.
        self.resolve_errors: list[str] = []

    def _resolve(self) -> None:
        """Auto-detect the best available LLM provider.

        Every provider branch used to end in a bare `except: pass`, so when a
        provider SDK was missing the whole subsystem degraded to "offline" with
        no indication of why. On this project that is the *default* state:
        `openai` and `groq` are imported here but declared in neither
        `pyproject.toml` nor the lockfile, so every branch raised
        `ModuleNotFoundError` and was swallowed. The result was a silently dead
        AI subsystem: callers got an "offline" engine and a rule-based fallback
        that answers natural language by guessing, with nothing in the logs.

        Failures are now collected into `resolve_errors` and warned about. The
        reason is a diagnostic, not a behavior change: resolution still ends at
        "offline" when no provider can be constructed.
        """
        if self._client is not None:
            return

        self.resolve_errors = []
        groq_key = os.environ.get("GROQ_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")

        # Local-first: prefer an already-running LM Studio server before any
        # cloud provider. No API key is needed and the model stays on-device.
        if self.provider == "auto":
            try:
                from openai import OpenAI
                local_client = OpenAI(
                    api_key="lm-studio",
                    base_url=os.environ.get("JARVIS_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1"),
                    timeout=120.0,
                    max_retries=0,
                )
                available = local_client.models.list().data
                if available:
                    self._client = local_client
                    self._resolved_provider = "lm-studio"
                    self._resolved_model = self.model or os.environ.get("JARVIS_LOCAL_MODEL") or available[0].id
                    return
            except Exception as exc:
                logger.debug("LM Studio local backend unavailable: %s", exc)

        def _fail(provider: str, exc: BaseException) -> None:
            self.resolve_errors.append(f"{provider}: {type(exc).__name__}: {exc}")
            logger.warning(
                "LLM provider %s unavailable (%s: %s)",
                provider,
                type(exc).__name__,
                exc,
            )

        # Try Gemini first if auto and GEMINI_API_KEY is available (verified working)
        if self.provider == "gemini" or (self.provider == "auto" and gemini_key):
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=gemini_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                self._resolved_provider = "gemini"
                self._resolved_model = self.model or os.environ.get("JARVIS_MODEL_NAME") or "gemini-flash-lite-latest"
                return
            except Exception as exc:
                _fail("gemini", exc)

        if self.provider == "groq" or (self.provider == "auto" and groq_key):
            try:
                from groq import Groq
                key = self.api_key or groq_key
                if key:
                    self._client = Groq(api_key=key)
                    self._resolved_provider = "groq"
                    self._resolved_model = self.model or "llama-3.3-70b-versatile"
                    return
            except Exception as exc:
                _fail("groq", exc)

        if self.provider == "openai" or self.provider == "auto":
            try:
                from openai import OpenAI
                key = self.api_key or openai_key
                if key:
                    kwargs: dict[str, Any] = {"api_key": key}
                    if self.base_url:
                        kwargs["base_url"] = self.base_url
                    self._client = OpenAI(**kwargs)
                    self._resolved_provider = "openai"
                    self._resolved_model = self.model or "gpt-4o-mini"
                    return
            except Exception as exc:
                _fail("openai", exc)

        # Use the JARVIS env vars as fallback
        if self.api_key and self.base_url:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                self._resolved_provider = "openai-compatible"
                self._resolved_model = self.model or os.environ.get("JARVIS_MODEL_NAME", "gpt-4o-mini")
                return
            except Exception as exc:
                _fail("openai-compatible", exc)

        if gemini_key or openai_key or groq_key:
            logger.warning(
                "No LLM provider could be constructed although an API key is "
                "present. The engine is OFFLINE; callers will fall back to "
                "rule-based behavior. Causes: %s",
                "; ".join(self.resolve_errors) or "no provider matched",
            )

        self._resolved_provider = "offline"
        self._resolved_model = "none"

    def plan_tool(
        self,
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Ask the configured model to choose a tool; return validated call arguments.

        Tool calls are proposals only. The caller enforces explicit user constraints,
        validates URLs/arguments, performs effects, and uses observed tool output as truth.
        """
        self._resolve()
        if self._client is None:
            return None
        response = self._client.chat.completions.create(
            model=self._resolved_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=min(self.max_tokens, 256),
        )
        choice = response.choices[0]
        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return None
        call = choice.message.tool_calls[0]
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            logger.warning("Local model returned malformed tool arguments for %s", call.function.name)
            return None
        if not isinstance(arguments, dict):
            return None
        return {"name": call.function.name, "arguments": arguments}

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from conversation messages."""
        self._resolve()

        if self._client is None:
            return self._offline_response(messages)

        try:
            response = self._client.chat.completions.create(
                model=self._resolved_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return str(content).strip() if content else ""
        except Exception as exc:
            # If primary provider failed, try falling back to OpenAI
            if self._resolved_provider != "openai":
                openai_key = os.environ.get("OPENAI_API_KEY", "")
                if openai_key:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=openai_key)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        content = response.choices[0].message.content
                        if content:
                            self._client = client
                            self._resolved_provider = "openai"
                            self._resolved_model = "gpt-4o-mini"
                            return str(content).strip()
                    except Exception:
                        pass
            print(f"  [LLM error: {exc}]")
            return self._offline_response(messages)

    def _offline_response(self, messages: list[dict[str, str]]) -> str:
        """Fallback when no LLM provider is available."""
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = msg["content"]
                break

        if not user_msg:
            return "I'm here, sir. How can I assist?"

        lower = user_msg.lower()
        if any(w in lower for w in ["who are you", "what are you", "your name"]):
            return "I am J.A.R.V.I.S. — Just A Rather Very Intelligent System. An autonomous AI companion engineered to assist you with all system and computational tasks, sir."
        if any(w in lower for w in ["hello", "hi", "hey"]):
            return "Good day, sir. How may I assist you?"
        if any(w in lower for w in ["time", "what time"]):
            return f"The current time is {time.strftime('%I:%M %p')}."
        if any(w in lower for w in ["date", "what day", "today"]):
            return f"Today is {time.strftime('%A, %B %d, %Y')}."
        if any(w in lower for w in ["weather"]):
            return "I'm afraid I don't have access to weather data in offline mode, sir."
        if any(w in lower for w in ["thank", "thanks"]):
            return "You're welcome, sir. Always happy to help."
        if any(w in lower for w in ["bye", "goodbye", "exit", "quit", "stop"]):
            return "Goodbye, sir. I'll be here when you need me."

        return (
            "I'm operating in offline mode without an LLM backend. "
            "Set GROQ_API_KEY or OPENAI_API_KEY for full intelligence. "
            f"You said: '{user_msg[:100]}'"
        )

    @property
    def is_online(self) -> bool:
        self._resolve()
        return self._client is not None

    @property
    def provider_info(self) -> str:
        self._resolve()
        return f"{self._resolved_provider} / {self._resolved_model}"


# ---------------------------------------------------------------------------
# The JARVIS Voice Assistant
# ---------------------------------------------------------------------------

class JarvisVoiceAssistant:
    """The complete real-time voice assistant loop.

    Wires: Microphone → VAD → STT → LLM → TTS → Speaker → loop

    Usage:
        assistant = JarvisVoiceAssistant()
        assistant.run()  # Blocks, runs the conversation loop
    """

    def __init__(
        self,
        *,
        stt_model: str = "base",
        use_google_stt: bool = False,
        tts_voice: str = DEFAULT_VOICE,
        llm_provider: str = "auto",
        llm_model: str | None = None,
        greeting: bool = True,
        verbose: bool = True,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.verbose = verbose
        self.greeting = greeting
        self.on_event = on_event
        self._browser_agent = None

        # Core engines
        self.stt = STTEngine(model=stt_model, use_google=use_google_stt)
        self.tts = NeuralTTSEngine(voice=tts_voice)
        self.llm = LLMEngine(provider=llm_provider, model=llm_model)
        self.memory = ConversationMemory()

        # Audio I/O
        self.speaker = RealSpeaker()

        # Skill Authoring Engine
        self.skill_creator = SkillCreator()

        # State
        self._running = False
        self._paused = False
        self._turn_count = 0

    def _log(self, msg: str) -> None:
        if self.on_event is not None:
            try:
                self.on_event(msg)
            except Exception:
                logger.debug("Voice event callback failed", exc_info=True)
        if self.verbose:
            try:
                print(msg, flush=True)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", None) or "ascii"
                clean = msg.encode(encoding, errors="replace").decode(encoding)
                print(clean, flush=True)

    def _speak(self, text: str) -> None:
        """Synthesize and play speech with human-like prosody and clean pronunciation."""
        cleaned = sanitize_speech_text(text)
        if not cleaned:
            return
        try:
            result = self.tts.synthesize(cleaned)
            self._log(f"  🔊 Speaking ({result.engine}, {result.duration_seconds:.1f}s)")
            self.speaker.play_wav(result.audio_bytes, blocking=True)
        except Exception as exc:
            self._log(f"  ⚠️  TTS failed: {exc}")

    def speak(self, text: str) -> None:
        """Public method to synthesize and play speech."""
        self._speak(text)

    def _think(self, user_text: str) -> str:
        """Generate an LLM response or perform explicitly requested safe actions."""
        # Minimal deterministic desktop actions do not depend on an LLM.
        # Only launch a named, known Windows application on a direct request.
        lower_text = user_text.lower().strip().rstrip(".!?")
        import subprocess
        import webbrowser

        # Browser actions are proposed by the local model through tool calling.
        # Deterministic validation preserves explicit user constraints before any side effect.
        normalized = re.sub(r"\s+", " ", lower_text).strip()
        browser_tools = [
            {"type": "function", "function": {
                "name": "browser_search",
                "description": "Search the browser for a query. Use for a web search request.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
            }},
            {"type": "function", "function": {
                "name": "browser_images",
                "description": "Open image search results for the query. Use when the user asks to view/find image results.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
            }},
            {"type": "function", "function": {
                "name": "browser_download",
                "description": "Download an image from the currently visible image results by 1-based ordinal. Use when the user asks to download/save an image. Do not use a default ordinal if the request has an explicit ordinal.",
                "parameters": {"type": "object", "properties": {"ordinal": {"type": "integer"}}, "required": ["ordinal"], "additionalProperties": False},
            }},
            {"type": "function", "function": {
                "name": "browser_inspect",
                "description": "Describe the currently open browser page and its visible content.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }},
            {"type": "function", "function": {
                "name": "browser_navigate",
                "description": "Navigate to a user-requested web URL.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False},
            }},
        ]
        # Record the user request exactly once, before model planning.
        self.memory.add("user", user_text)
        try:
            # Keep tool selection focused: the full telemetry/persona prompt can
            # overwhelm small local models and make them ignore available tools.
            # Conversation history remains available to response generation and the
            # follow-up planner; this first decision needs only the current request.
            planner_messages = [
                {
                    "role": "system",
                    "content": (
                        "Choose the best available tool for the user's current request. "
                        "Call browser_search for web searches, browser_images to find/view "
                        "images, browser_download to save an image from already-open results, "
                        "browser_inspect to describe the current page, and browser_navigate "
                        "to open a URL the user asks to visit. For a request combining image "
                        "search and download, start with browser_images; the application will "
                        "show you the results before asking you to choose the download action. "
                        "Use no tool only when the request needs no listed browser action."
                    ),
                },
                {"role": "user", "content": user_text},
            ]
            plan = self.llm.plan_tool(messages=planner_messages, tools=browser_tools)
        except Exception as exc:
            self._log(f"  [Tool planner unavailable: {type(exc).__name__}: {exc}]")
            plan = None

        # A model proposal is never permission to violate explicit task constraints.
        ordinal_words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1}
        explicit_ordinal = None
        ordinal_token = r"(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)"
        ordinal_match = re.search(
            rf"\b{ordinal_token}\b\s+(?:number\s+)?(?:image|photo|picture)\b|\b(?:image|photo|picture)\s+(?:number\s+)?{ordinal_token}\b",
            normalized,
        )
        if ordinal_match:
            token = next((group for group in ordinal_match.groups() if group), "")
            explicit_ordinal = int(token) if token.isdigit() else ordinal_words.get(token)

        if plan and plan.get("name") in {"browser_search", "browser_images", "browser_download", "browser_inspect", "browser_navigate"}:
            name, args = plan["name"], plan["arguments"]
            try:
                if name == "browser_download" and explicit_ordinal is not None:
                    proposed = args.get("ordinal")
                    if proposed not in (None, explicit_ordinal):
                        return f"I understood that you requested image {explicit_ordinal}, but the model proposed image {proposed}. I won't download a different image."
                    args["ordinal"] = explicit_ordinal
                if name == "browser_download" and explicit_ordinal == -1:
                    return "I can't safely identify the last image from the current results yet. Please ask for a numbered image, or let me inspect the results first."
                if self._browser_agent is None:
                    from .browser_agent import BrowserAgent
                    self._browser_agent = BrowserAgent()
                if name == "browser_search":
                    reply = self._browser_agent.search(str(args["query"]).strip())
                elif name == "browser_images":
                    reply = self._browser_agent.search(str(args["query"]).strip())
                    if "opened search results" in reply.lower() or "opened bing results" in reply.lower():
                        reply = self._browser_agent.click_images_tab()
                    # Compound requests continue through the model tool loop. The model
                    # observes the actual browser result first; any later download still
                    # passes through explicit-ordinal validation above.
                    wants_download = bool(re.search(r"\b(download|save)\b", normalized) and re.search(r"\b(image|photo|picture)\b", normalized))
                    if wants_download:
                        followup_messages = self.memory.get_messages() + [
                            {"role": "assistant", "content": f"Browser tool result: {reply}"},
                            {"role": "user", "content": "Continue the original request using the observed browser state. If the user requested an image download, call browser_download with the exact requested ordinal. Do not repeat search unless required."},
                        ]
                        followup = self.llm.plan_tool(messages=followup_messages, tools=browser_tools)
                        if followup and followup.get("name") == "browser_download":
                            follow_args = followup.get("arguments", {})
                            if explicit_ordinal is not None:
                                # The user's explicit ordinal is authoritative, even when a
                                # weaker model proposes a different index on the follow-up.
                                follow_args["ordinal"] = explicit_ordinal
                            if explicit_ordinal == -1:
                                return "Image results are open, but I can't safely identify the last image yet. Please request a numbered image."
                            reply = self._browser_agent.download_visible_image(int(follow_args.get("ordinal", 1)))
                        elif not followup:
                            reply = f"{reply} I opened the image results, but the local model did not produce a safe download action."
                elif name == "browser_download":
                    if explicit_ordinal is not None and args.get("ordinal") != explicit_ordinal:
                        return "The requested image number could not be preserved; nothing was downloaded."
                    reply = self._browser_agent.download_visible_image(int(args.get("ordinal", 1)))
                elif name == "browser_inspect":
                    reply = self._browser_agent.inspect_page()
                else:
                    url = str(args.get("url", "")).strip()
                    if not re.match(r"^https?://", url, re.I):
                        return "I can only navigate to a valid HTTP or HTTPS URL. Nothing was opened."
                    reply = self._browser_agent.navigate(url)
            except Exception as exc:
                reply = f"Browser action failed: {exc}"
            self.memory.add("assistant", reply)
            return reply

        # General Windows folder/file-explorer intent: resolve the target dynamically.
        if "explorer" in normalized or "file explorer" in normalized:
            if self.memory.turn_count and self.memory._turns[-1].role == "user" and self.memory._turns[-1].content == user_text:
                self.memory._turns.pop()
            import os
            from pathlib import Path
            target = None
            folder_match = re.search(r"(?:open|show|go to|navigate to)\s+(.+?)\s+(?:in|using)\s+(?:(?:my|windows) )?file explorer$", normalized)
            if folder_match:
                requested = folder_match.group(1).strip().strip("\\\"'")
                known_folders = {
                    "downloads": Path.home() / "Downloads",
                    "documents": Path.home() / "Documents",
                    "desktop": Path.home() / "Desktop",
                    "pictures": Path.home() / "Pictures",
                    "music": Path.home() / "Music",
                    "videos": Path.home() / "Videos",
                    "home": Path.home(),
                    "this pc": None,
                }
                target = known_folders.get(requested.lower())
                if target is None and requested.lower() not in known_folders:
                    target = Path(os.path.expandvars(os.path.expanduser(requested)))
                if target is not None and not target.exists():
                    reply = f"The requested location doesn't exist: {target}"
                else:
                    try:
                        subprocess.Popen(["explorer.exe", str(target)] if target else ["explorer.exe"])
                        reply = f"Opened File Explorer{f' at {target}' if target else ''}."
                    except OSError as exc:
                        reply = f"Could not open File Explorer: {exc}"
            else:
                try:
                    subprocess.Popen(["explorer.exe"])
                    reply = "Opened Windows File Explorer."
                except OSError as exc:
                    reply = f"Could not open File Explorer: {exc}"
            self.memory.add("user", user_text)
            self.memory.add("assistant", reply)
            return reply

        # Explicit allow-listed desktop commands execute independently of the LLM.
        app_commands = {
            "open notepad": (["notepad.exe"], "Notepad"),
            "launch notepad": (["notepad.exe"], "Notepad"),
            "start notepad": (["notepad.exe"], "Notepad"),
            "open firefox": (["cmd", "/c", "start", "", "firefox"], "Firefox"),
            "launch firefox": (["cmd", "/c", "start", "", "firefox"], "Firefox"),
            "start firefox": (["cmd", "/c", "start", "", "firefox"], "Firefox"),
            "open spotify": (["cmd", "/c", "start", "", "spotify:"], "Spotify"),
            "launch spotify": (["cmd", "/c", "start", "", "spotify:"], "Spotify"),
        }
        if lower_text in app_commands:
            if self.memory.turn_count and self.memory._turns[-1].role == "user" and self.memory._turns[-1].content == user_text:
                self.memory._turns.pop()
            command, app_name = app_commands[lower_text]
            try:
                subprocess.Popen(command, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                reply = f"Opening {app_name} now, sir."
            except OSError as exc:
                reply = f"I couldn't open {app_name}: {exc}"
            self.memory.add("user", user_text)
            self.memory.add("assistant", reply)
            return reply

        website_commands = {
            "open makaut website": "https://makaut.ac.in/",
            "open the makaut website": "https://makaut.ac.in/",
            "open macout website": "https://makaut.ac.in/",
            "open the macout website": "https://makaut.ac.in/",
            "open spotify website": "https://open.spotify.com/",
        }
        if lower_text in website_commands:
            url = website_commands[lower_text]
            opened = webbrowser.open(url)
            reply = f"Opened {url} in your default browser." if opened else f"Couldn't confirm browser launch for {url}."
            self.memory.add("assistant", reply)
            return reply

        if lower_text in {"open firefox and open makaut website", "open firefox and open the makaut website", "open firefox and open macout website", "open firefox and open the macout website"}:
            try:
                subprocess.Popen(["cmd", "/c", "start", "", "firefox"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError:
                pass
            opened = webbrowser.open("https://makaut.ac.in/")
            reply = "Requested Firefox launch and opened the MAKAUT website in the default browser." if opened else "Firefox launch requested, but browser navigation could not be confirmed."
            self.memory.add("assistant", reply)
            return reply

        # Check if user instructed JARVIS to create a new skill
        if is_skill_creation_intent(user_text):
            result = self.skill_creator.create_skill(user_prompt=user_text, llm_engine=self.llm)
            self.memory.add("assistant", result.message)
            return result.message

        # The turn is already in memory; do not append it a second time.
        messages = self.memory.get_messages()
        response = self.llm.generate(messages)
        if response:
            self.memory.add("assistant", response)
        return response

    def run(self) -> None:
        """Run the main conversation loop (blocking).

        Uses speech_recognition's built-in mic listener for reliable
        audio capture with automatic energy threshold adjustment.
        """
        self._running = True
        self._log("")
        self._log("=" * 60)
        self._log("  ╔═══════════════════════════════════════╗")
        self._log("  ║      J.A.R.V.I.S. Voice Assistant     ║")
        self._log("  ║   Just A Rather Very Intelligent System║")
        self._log("  ╚═══════════════════════════════════════╝")
        self._log("=" * 60)
        self._log(f"  STT: Whisper ({self.stt.model})")
        self._log(f"  TTS: {self.tts.voice}")
        self._log(f"  LLM: {self.llm.provider_info}")
        self._log(f"  Status: {'Online' if self.llm.is_online else 'Offline (set GROQ_API_KEY)'}")
        self._log("=" * 60)
        self._log("  Say something or press Ctrl+C to exit.")
        self._log("")

        # Greeting
        if self.greeting:
            self._log(f"  🤖 JARVIS: {JARVIS_GREETING}")
            self._speak(JARVIS_GREETING)

        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        try:
            with sr.Microphone(sample_rate=DEFAULT_SAMPLE_RATE) as source:
                self._log("  🎤 Calibrating microphone...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                self._log(f"  🎤 Microphone ready (energy threshold: {recognizer.energy_threshold:.0f})")
                self._log("")

                while self._running:
                    try:
                        self._log("  👂 Listening...")

                        # Listen for speech (blocks until speech detected + silence)
                        audio = recognizer.listen(
                            source,
                            timeout=None,  # Wait indefinitely for speech
                            phrase_time_limit=30,  # Max 30s per utterance
                        )

                        self._turn_count += 1
                        self._log(f"  📝 Processing turn #{self._turn_count}...")

                        # STT: Convert speech to text
                        try:
                            transcript = self.stt.transcribe_audio(audio)
                        except sr.UnknownValueError:
                            self._log("  ❓ Couldn't understand. Try again.")
                            continue
                        except sr.RequestError as exc:
                            self._log(f"  ⚠️  STT error: {exc}")
                            continue

                        if not transcript:
                            self._log("  (empty transcript, skipping)")
                            continue

                        self._log(f"  🗣️  You: {transcript}")

                        # Check for exit commands
                        lower = transcript.lower().strip()
                        if lower in ("exit", "quit", "stop", "goodbye", "bye", "shut down",
                                     "jarvis stop", "jarvis exit", "jarvis quit"):
                            farewell = "Goodbye, sir. Shutting down. I'll be here when you need me."
                            self._log(f"  🤖 JARVIS: {farewell}")
                            self._speak(farewell)
                            self._running = False
                            break

                        # Check for pause/resume
                        if lower in ("pause", "jarvis pause", "mute", "jarvis mute"):
                            self._log("  ⏸️  Paused. Say 'resume' to continue.")
                            self._speak("Pausing, sir. Say resume when you're ready.")
                            self._paused = True
                            continue

                        if lower in ("resume", "jarvis resume", "unmute", "jarvis unmute"):
                            self._paused = False
                            self._log("  ▶️  Resumed.")
                            self._speak("I'm back, sir. What do you need?")
                            continue

                        if self._paused:
                            continue

                        # LLM: Generate response
                        self._log("  🧠 Thinking...")
                        response = self._think(transcript)

                        if not response:
                            response = "I'm sorry, I didn't have a response for that."

                        self._log(f"  🤖 JARVIS: {response}")

                        # TTS: Speak the response
                        self._speak(response)
                        self._log("")

                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        self._log(f"  ⚠️  Turn error: {exc}")
                        time.sleep(0.5)

        except KeyboardInterrupt:
            self._log("")
            self._log("  Ctrl+C received. Shutting down...")
            self._speak("Shutting down. Goodbye, sir.")
        finally:
            self._running = False
            # Playwright owns a Node driver process; close it before the voice worker exits
            # so its event pipe cannot be left writing to a closed Python process.
            if self._browser_agent is not None:
                try:
                    self._browser_agent.close()
                except Exception as exc:
                    self._log(f"  Browser cleanup warning: {exc}")
                finally:
                    self._browser_agent = None
            self.speaker.stop_playback()
            self._log("  JARVIS offline.")

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._running = False

    def single_turn(self, text: str) -> str:
        """Process a single text turn (for testing/programmatic use)."""
        response = self._think(text)
        return response


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: `python -m jarvis.multimodal.jarvis_voice`."""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Voice Assistant")
    parser.add_argument("--stt-model", default="base", help="Whisper model size (tiny/base/small/medium)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Edge TTS voice name")
    parser.add_argument("--llm", default="auto", help="LLM provider (auto/groq/openai)")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--no-greeting", action="store_true", help="Skip greeting")
    parser.add_argument("--google-stt", action="store_true", help="Use Google STT instead of Whisper")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--list-voices", action="store_true", help="List available Edge TTS voices")

    args = parser.parse_args()

    if args.list_voices:
        print("Available JARVIS voices:")
        for name, voice_id in EDGE_VOICES.items():
            marker = " ← default" if voice_id == DEFAULT_VOICE else ""
            print(f"  {name:20s} = {voice_id}{marker}")
        print(f"\n  Current default: {DEFAULT_VOICE}")
        print("  Set with --voice <voice-id>")
        return

    assistant = JarvisVoiceAssistant(
        stt_model=args.stt_model,
        use_google_stt=args.google_stt,
        tts_voice=args.voice,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        greeting=not args.no_greeting,
        verbose=not args.quiet,
    )
    assistant.run()


if __name__ == "__main__":
    main()
