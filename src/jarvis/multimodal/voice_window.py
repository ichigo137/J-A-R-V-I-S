"""Small desktop conversation window for the continuous JARVIS voice loop.

Run with: python -m jarvis.multimodal.voice_window
The UI runs on Tk's main thread; microphone/STT/LLM/TTS work stays on a
background thread. The assistant's existing pipeline and memory are reused.
"""
from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from .jarvis_voice import JarvisVoiceAssistant, DEFAULT_VOICE


class VoiceWindow:
    def __init__(self, *, stt_model: str = "base", voice: str = DEFAULT_VOICE,
                 llm: str = "auto", llm_model: str | None = None,
                 google_stt: bool = False, greeting: bool = True) -> None:
        self.root = tk.Tk()
        self.root.title("JARVIS — Voice Conversation")
        self.root.geometry("760x620")
        self.root.minsize(520, 400)
        self.events: queue.Queue[str] = queue.Queue()
        self.assistant: JarvisVoiceAssistant | None = None
        self.worker: threading.Thread | None = None
        self.options = dict(stt_model=stt_model, tts_voice=voice,
                            llm_provider=llm, llm_model=llm_model,
                            use_google_stt=google_stt, greeting=greeting,
                            verbose=False, on_event=self.events.put)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="JARVIS", font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)
        self.status = tk.StringVar(value="Offline")
        ttk.Label(header, textvariable=self.status).pack(side=tk.RIGHT)
        self.chat = scrolledtext.ScrolledText(
            outer, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10),
            padx=10, pady=10
        )
        self.chat.pack(fill=tk.BOTH, expand=True, pady=(12, 10))
        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X)
        self.start_button = ttk.Button(controls, text="Start listening", command=self.start)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(controls, text="Continuous microphone · local Whisper by default").pack(side=tk.RIGHT)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._append("JARVIS", "Ready. Start listening to begin a hands-free conversation.")
        self.root.after(100, self._drain_events)

    def _append(self, speaker: str, text: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{speaker}: {text.strip()}\n\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def _on_event(self, message: str) -> None:
        # Called from the voice worker; queue consumed by the Tk thread.
        self.events.put(message)

    def _drain_events(self) -> None:
        while True:
            try:
                message = self.events.get_nowait()
            except queue.Empty:
                break
            clean = message.strip()
            if not clean:
                continue
            if "Listening..." in clean:
                self.status.set("Listening")
            elif "Thinking..." in clean:
                self.status.set("Thinking")
            elif "offline." in clean.lower():
                self.status.set("Offline")
                self.start_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)
            elif "Microphone ready" in clean:
                self.status.set("Listening")
            if "You:" in clean:
                self._append("You", clean.split("You:", 1)[1])
            elif "JARVIS:" in clean:
                self._append("JARVIS", clean.split("JARVIS:", 1)[1])
            elif "error" in clean.lower() or "couldn't understand" in clean.lower():
                self._append("Status", clean)
        self.root.after(100, self._drain_events)

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            self.assistant = JarvisVoiceAssistant(**self.options)
        except Exception as exc:
            self._append("Error", str(exc))
            return
        self.status.set("Starting")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.worker = threading.Thread(target=self._run_assistant, daemon=True,
                                       name="jarvis-voice-loop")
        self.worker.start()

    def _run_assistant(self) -> None:
        try:
            assert self.assistant is not None
            self.assistant.run()
        except Exception as exc:
            self.events.put(f"Voice startup error: {exc}")
            self.events.put("JARVIS offline.")
        finally:
            self.events.put("JARVIS offline.")

    def stop(self) -> None:
        if self.assistant is not None:
            self.assistant.stop()
        self.status.set("Stopping…")
        self.stop_button.configure(state=tk.DISABLED)

    def close(self) -> None:
        self.stop()
        self.root.after(150, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS desktop voice conversation window")
    parser.add_argument("--stt-model", default="base", help="Whisper model size")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="TTS voice name")
    parser.add_argument("--llm", default="auto", help="LLM provider (auto/groq/openai)")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--google-stt", action="store_true", help="Use cloud Google speech recognition")
    parser.add_argument("--no-greeting", action="store_true", help="Skip spoken greeting")
    args = parser.parse_args()
    VoiceWindow(stt_model=args.stt_model, voice=args.voice, llm=args.llm,
                llm_model=args.llm_model, google_stt=args.google_stt,
                greeting=not args.no_greeting).run()


if __name__ == "__main__":
    main()
