from __future__ import annotations

"""The voice turn - the ONE owner of one spoken turn.

    stt -> turn-taking FSM -> answer -> tts

`VoiceTurn.run` is the whole turn and nothing else: it transcribes (or says why it
could not), feeds the real `StreamingVoiceLoop` FSM the real samples (or declares
that nothing was captured), answers from memory (or hands the question to the
caller's `answerer`), synthesizes and MEASURES the audio before claiming it spoke,
and returns a `VoiceTurnReport`.

It owns no configuration: the engines and their honest labels come from
`jarvis.live_boot` (`real - ...` / `seam - ...`), the wording of every line comes
from `jarvis.live_report`, the offline answer path and the component listing are
handed in by the composition root (`jarvis.live`), and the per-frame VAD facts and
the WAV measurement come from `jarvis.live_boot`'s real parsers.

Never faked here: with no STT engine bound a supplied audio file is NOT
transcribed (no mock transcript is fabricated), and a synthesis that returns no
bytes, a non-WAV payload or a silent WAV is reported as a FAILED synthesis rather
than spoken.
"""

import asyncio
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable

from .kernel.event_log import EventLog, new_ulid
from .live_boot import (
    AUDIO_DIRNAME,
    VAD_FRAME_MS,
    VAD_RMS_THRESHOLD,
    _wav_facts,
    _wav_vad_frames,
)
from .live_report import (
    VoiceTurnReport,
    answer_line,
    stt_seam_line,
    stt_transcribed_line,
    stt_typed_line,
    tts_failed_line,
    tts_seam_line,
    tts_skipped_line,
    tts_wrote_line,
    vad_declared_line,
    vad_line,
    vad_real_line,
    voice_answer_path,
    voice_loop_line,
)
from .multimodal.audio_io import (
    DEFAULT_SPEECH_FRAMES,
    RealMicrophone,
    RealSpeaker,
    find_default_input_device,
)
from .multimodal.streaming import StreamingVoiceLoop


class VoiceTurn:
    """Runs one voice turn over the engines this runtime really has bound.

    `stt`/`tts` are None when no engine bound (so a turn can never be fed a
    deterministic default runner and reported as perception); `offline_answer` is
    the deterministic memory answer used when the caller supplies no `answerer`;
    `components` is the composition root's honest component listing.
    """

    def __init__(
        self,
        *,
        log: EventLog,
        home: Path,
        stt: Any,
        stt_engine: str,
        tts: Any,
        tts_engine: str,
        offline_answer: Callable[[str], Any],
        components: Callable[..., tuple[str, ...]],
    ) -> None:
        self._log = log
        self._home = home
        self._stt = stt
        self._stt_engine = stt_engine
        self._tts = tts
        self._tts_engine = tts_engine
        self._offline_answer = offline_answer
        self._components = components

    def _capture_microphone(self, lines: list[str]) -> Path | None:
        """Capture one bounded utterance from the physical default microphone."""
        device = find_default_input_device()
        if device is None:
            lines.append("microphone: FAILED - no default input device is available")
            return None
        mic = RealMicrophone(device_index=device)
        # Keep a short pre-roll so the first consonants/syllables are not lost
        # while VAD decides that the user has started speaking.
        pre_roll: deque[bytes] = deque(maxlen=12)  # ~360 ms at 30 ms/frame
        captured: list[bytes] = []
        saw_speech = False
        speech_frames = 0
        consecutive_speech = 0
        silence_frames = 0
        consumed_frames = 0
        max_capture_frames = int(12.0 * 1000 / VAD_FRAME_MS)
        deadline = time.monotonic() + 12.0
        try:
            mic.start()
            lines.append(f"microphone: REAL capture started (device {device})")
            while time.monotonic() < deadline:
                frame = mic.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                consumed_frames += 1
                if consumed_frames > max_capture_frames:
                    break
                if not saw_speech:
                    pre_roll.append(frame.data)
                if frame.is_speech:
                    consecutive_speech += 1
                    if not saw_speech and consecutive_speech >= DEFAULT_SPEECH_FRAMES:
                        captured.extend(pre_roll)
                        pre_roll.clear()
                        saw_speech = True
                    if saw_speech:
                        captured.append(frame.data)
                    speech_frames += 1
                    silence_frames = 0
                elif saw_speech:
                    consecutive_speech = 0
                    captured.append(frame.data)
                    silence_frames += 1
                    # ~750 ms of trailing silence gives short utterances and
                    # Whisper a little more room than the old ~450 ms cutoff.
                    if silence_frames >= 25:
                        break
                else:
                    consecutive_speech = 0
        except Exception as exc:
            lines.append(f"microphone: FAILED - {type(exc).__name__}: {exc}")
            return None
        finally:
            mic.stop()
        if not saw_speech or not captured:
            lines.append("microphone: no speech captured before the 12s timeout")
            return None
        target_dir = self._home / AUDIO_DIRNAME
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"mic-{new_ulid()}.wav"
        with wave.open(str(target), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"".join(captured))
        lines.append(f"microphone: captured {len(captured)} frames to {target}")
        return target

    def run(
        self,
        utterance: str,
        *,
        audio: Path | str | None = None,
        microphone: bool = False,
        answerer: Callable[[str], Any] | None = None,
    ) -> VoiceTurnReport:
        lines: list[str] = []
        transcript = (utterance or "").strip()
        source = "typed utterance"
        frames: list[tuple[bytes, bool]] | None = None

        if microphone and audio is not None:
            lines.append("microphone: ignored because --audio was supplied")
        if microphone and audio is None and not transcript:
            audio = self._capture_microphone(lines)
            if audio is None:
                source = "physical microphone (no speech captured)"

        if audio is not None:
            path = Path(audio)
            try:
                raw_audio = path.read_bytes()
            except OSError:
                raw_audio = b""
            # real frames, real per-frame energy: the FSM is fed the samples
            frames = _wav_vad_frames(raw_audio) if raw_audio else None
            if self._stt is None:
                lines.append(stt_seam_line(self._stt_engine, path.name))
            else:
                try:
                    spoken = asyncio.run(
                        self._stt.invoke(
                            "audio.transcribe",
                            "1.0.0",
                            {"audio_path": str(path), "format": path.suffix.lstrip(".") or "wav"},
                        )
                    )
                except Exception as exc:
                    lines.append(
                        f"stt: {self._stt_engine} FAILED - {type(exc).__name__}: {exc}; "
                        "no transcript was fabricated"
                    )
                    transcript = ""
                    source = f"STT failure for {path.name}"
                else:
                    transcript = str(spoken.get("text", "")).strip()
                    source = f"transcribed from {path.name} by {spoken.get('engine', 'engine')}"
                    lines.append(stt_transcribed_line(self._stt_engine, transcript))
        else:
            lines.append(stt_typed_line(self._stt_engine))

        # the real turn-taking FSM: every transition is a real audit event
        states: list[str] = []
        loop = StreamingVoiceLoop(
            event_log=self._log,
            on_state_change=lambda _old, new: states.append(new.value),
        )
        loop.start_listening()
        if frames:
            speech_frames = sum(1 for _frame, active in frames if active)
            for frame, active in frames:
                loop.feed_audio_frame(frame, vad_active=active)
            vad = vad_real_line(
                frame_count=len(frames),
                frame_ms=VAD_FRAME_MS,
                audio_name=Path(audio).name,
                speech_frames=speech_frames,
                rms_threshold=VAD_RMS_THRESHOLD,
            )
        else:
            # no frames to measure: say so instead of implying a capture
            loop.feed_audio_frame(b"", vad_active=bool(transcript))
            vad = vad_declared_line()
        for _ in range(loop.silence_threshold_frames):
            loop.feed_audio_frame(b"", vad_active=False)

        if not transcript:
            lines.append("answer: skipped - no transcript available")
            return VoiceTurnReport(
                utterance="",
                transcript_source=source,
                stt=self._stt_engine,
                answered=False,
                answer="",
                confidence=0.0,
                used_model=False,
                tts=self._tts_engine,
                audio_out=None,
                states=tuple(states),
                lines=tuple(lines),
                components=tuple(),
                vad=vad,
                audio_seconds=0.0,
                audio_sample_rate=0,
                audio_peak=0.0,
                audio_non_silent=False,
            )

        result = (
            answerer(transcript) if answerer is not None else self._offline_answer(transcript)
        )
        answered = bool(getattr(result, "answered", False))
        answer = getattr(result, "answer", "") if answered else ""
        used_model = bool(getattr(result, "used_model", False))
        provider_id = str(getattr(result, "provider_id", "gateway"))
        confidence = float(getattr(result, "confidence", 0.0))
        loop.start_assistant_speaking()

        audio_out: str | None = None
        audio_seconds = 0.0
        audio_rate = 0
        audio_peak = 0.0
        audio_non_silent = False
        if not answered:
            lines.append(tts_skipped_line(self._tts_engine))
        elif self._tts is None:
            lines.append(tts_seam_line(self._tts_engine))
        else:
            try:
                spoken_out = asyncio.run(
                    self._tts.invoke(
                        "audio.synthesize", "1.0.0", {"text": answer, "format": "wav"}
                    )
                )
                payload = bytes(spoken_out["audio_bytes"])
                if not payload:
                    raise ValueError("the engine returned no bytes")
                # measured, never assumed: a payload that is not a real WAV, or a
                # WAV that carries only silence, is a FAILED synthesis. Success is
                # not evidence here any more than `exit 0` is on the worker side.
                facts = _wav_facts(payload)
                if not facts["non_silent"]:
                    raise ValueError("the engine wrote a silent WAV (peak is zero)")
            except Exception as exc:  # a synthesis fault is reported, never faked
                lines.append(tts_failed_line(self._tts_engine, exc))
            else:
                target_dir = self._home / AUDIO_DIRNAME
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"turn-{new_ulid()}.wav"
                target.write_bytes(payload)
                audio_out = str(target)
                audio_seconds = facts["duration_seconds"]
                audio_rate = facts["sample_rate"]
                audio_peak = facts["peak"]
                audio_non_silent = facts["non_silent"]
                lines.append(
                    tts_wrote_line(self._tts_engine, str(target), facts, len(payload))
                )
                if microphone:
                    # Physical microphone turns are interactive: play the verified
                    # synthesis through the real default output device.
                    try:
                        speaker = RealSpeaker()
                        speaker.play_wav(payload, blocking=True)
                        lines.append("speaker: REAL default output playback completed")
                    except Exception as exc:
                        lines.append(
                            f"speaker: FAILED - {type(exc).__name__}: {exc}; "
                            "the synthesized WAV was still written and measured"
                        )
        loop.finish_assistant_speaking()

        lines.append(voice_loop_line(states))
        lines.append(vad_line(vad))
        lines.append(
            answer_line(
                answered=answered,
                answer=answer,
                used_model=used_model,
                provider_id=provider_id,
                confidence=confidence,
            )
        )
        return VoiceTurnReport(
            utterance=transcript,
            transcript_source=source,
            stt=self._stt_engine,
            answered=answered,
            answer=answer,
            confidence=confidence,
            used_model=used_model,
            tts=self._tts_engine,
            audio_out=audio_out,
            states=tuple(states),
            lines=tuple(lines),
            vad=vad,
            audio_seconds=audio_seconds,
            audio_sample_rate=audio_rate,
            audio_peak=audio_peak,
            audio_non_silent=audio_non_silent,
            components=self._components(
                mission=False,
                answer_path=voice_answer_path(used_model, provider_id),
            ),
        )


__all__ = ["VoiceTurn"]
