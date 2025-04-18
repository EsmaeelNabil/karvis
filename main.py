#!/usr/bin/env python3
"""
Main loop: real-time speech → LLM → TTS.
Now using Rich Console for chat panels.
"""

import os
import sys
import threading
import time

from rich.console import Console
from rich.panel import Panel

from llm_completion import LLMCompletion
from kokoro_tts import KokoroTTS
from moonshine import RealTimeTranscriber

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# --- Configuration ---
TRANSCRIBER_MODEL = "moonshine/base"
SILENCE_TIMEOUT_MS = 300
# LLM_MODEL = "gpt-4o-mini"
LLM_MODEL = "cogito:3b"
USE_LOCAL_LLM = True
TTS_MODEL_ID = "prince-canuma/Kokoro-82M"
TTS_VOICE = "af_bella"

# --- Rich console instance ---
console = Console()


def main_loop():
    # Initialize transcriber, LLM, and TTS
    transcriber = RealTimeTranscriber(
        model_name=TRANSCRIBER_MODEL,
        vad_threshold=0.5,
        vad_min_silence_ms=SILENCE_TIMEOUT_MS,
    )
    llm = LLMCompletion(model=LLM_MODEL, local=USE_LOCAL_LLM)
    tts = KokoroTTS(kokoro_model_id=TTS_MODEL_ID, lang_code="b", voice=TTS_VOICE)
    if tts.pipeline is None:
        console.log("Warning: TTS pipeline unavailable; audio replies disabled.")

    stop_event = threading.Event()

    def listen_for_quit():
        console.log("Press 'q' + Enter to quit.")
        while not stop_event.is_set():
            if sys.stdin.isatty():
                cmd = input().strip().lower()
                if cmd == "q":
                    stop_event.set()
                    break
            else:
                time.sleep(0.5)

    listener = threading.Thread(target=listen_for_quit, daemon=True)
    listener.start()

    try:
        console.log("Speak now…")
        for sentence in transcriber.start():
            if stop_event.is_set():
                break
            if not sentence:
                continue

            # Display user message in a cyan panel
            console.print(
                Panel(sentence, title="You", border_style="cyan", expand=False)
            )  # :contentReference[oaicite:5]{index=5}

            # Generate and display assistant reply
            reply = llm.complete(sentence)
            console.print(
                Panel(reply, title="Assistant", border_style="green", expand=False)
            )  #

            # Pause mic while speaking
            try:
                if transcriber._stream:
                    transcriber._stream.stop()
            except Exception:
                console.log("Warning: failed to pause stream.")

            # Play TTS
            if reply != "[Error generating response]" and tts.pipeline:
                tts.play_text(reply)
            else:
                console.log("Skipping TTS…")

            # Resume mic
            try:
                if transcriber._stream:
                    transcriber._stream.start()
            except Exception:
                console.log("Warning: failed to resume stream.")

    except KeyboardInterrupt:
        console.log("Interrupted by user.")
    finally:
        stop_event.set()
        transcriber.stop()
        console.log("Exited main loop. Goodbye!")


if __name__ == "__main__":
    main_loop()
