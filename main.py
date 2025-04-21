#!/usr/bin/env python3
"""
Main loop: real-time speech → LLM → TTS.
Now using Rich Console for chat panels.
"""

import os
import sys
import threading
import time
import argparse

from rich.console import Console
from rich.panel import Panel

from llm_completion import LLMCompletion
from kokoro_tts import KokoroTTS
from moonshine import RealTimeTranscriber

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time speech → LLM → TTS loop")
    p.add_argument(
        "--transcriber-model",
        type=str,
        default="moonshine/base",
        help="name of the transcription model",
    )
    p.add_argument(
        "--silence-timeout-ms",
        type=int,
        default=300,
        help="milliseconds of silence before treating as end of utterance",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="cogito:3b",
        help="LLM model to use (e.g. gpt-4o-mini, cogito:3b)",
    )
    p.add_argument(
        "--use-local-llm",
        dest="use_local_llm",
        action="store_true",
        help="run LLM locally instead of via API",
    )
    p.add_argument(
        "--no-local-llm",
        dest="use_local_llm",
        action="store_false",
        help="disable local LLM (i.e. force API usage)",
    )
    p.set_defaults(use_local_llm=True)
    p.add_argument(
        "--tts-model-id",
        type=str,
        default="prince-canuma/Kokoro-82M",
        help="HuggingFace model ID for Kokoro TTS",
    )
    p.add_argument(
        "--tts-voice",
        type=str,
        default="af_bella",
        help="voice identifier for Kokoro TTS",
    )
    return p.parse_args()


def main_loop(args):
    # Initialize transcriber, LLM, and TTS
    transcriber = RealTimeTranscriber(
        model_name=args.transcriber_model,
        vad_threshold=0.5,
        vad_min_silence_ms=args.silence_timeout_ms,
    )
    llm = LLMCompletion(model=args.llm_model, local=args.use_local_llm)
    tts = KokoroTTS(
        kokoro_model_id=args.tts_model_id, lang_code="b", voice=args.tts_voice
    )
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

            console.print(
                Panel(sentence, title="You", border_style="cyan", expand=False)
            )

            reply = llm.complete(sentence)
            console.print(
                Panel(reply, title="Assistant", border_style="green", expand=False)
            )

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
    console = Console()
    args = parse_args()
    main_loop(args)
