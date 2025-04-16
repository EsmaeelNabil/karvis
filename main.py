
import sys
import threading
import time
from chat_tts import ChatTTS

# Import real-time transcriber
try:
    from transcriber_module import RealTimeTranscriber
except ImportError:
    print("Error: Could not import RealTimeTranscriber.", file=sys.stderr)
    print(
        "Make sure 'transcriber_module.py' is in the same directory or Python path.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Configuration ---
TRANSCRIBER_MODEL = "mlx-community/whisper-tiny-mlx"
SILENCE_TIMEOUT_MS = 1500

# Instantiate ChatTTS once (loads models)
assistant = ChatTTS()


# --- Transcription and Chat Loop ---
def main_loop():
    # Start real-time transcription
    transcriber = RealTimeTranscriber(
        model_name_or_path=TRANSCRIBER_MODEL, vad_silence_timeout_ms=SILENCE_TIMEOUT_MS
    )

    stop_event = threading.Event()

    def listener():
        print("*** Press 'q' then Enter to quit ***")
        while not stop_event.is_set():
            try:
                if sys.stdin.isatty():
                    inp = input()
                    if inp.strip().lower() == "q":
                        stop_event.set()
                        break
                else:
                    time.sleep(0.2)
            except Exception:
                stop_event.set()

    # Launch quit listener
    threading.Thread(target=listener, daemon=True).start()

    try:
        print("Starting transcription. Speak now...", file=sys.stderr)
        for sentence in transcriber.start():
            if stop_event.is_set():
                break

            # Generate response text only
            reply = assistant.generate_text(f"USER: {sentence}\nASSISTANT:")
            print(f"You: {sentence}")
            print(f"Assistant: {reply}")

            # Pause audio stream to avoid capturing model's voice
            try:
                if hasattr(transcriber, "_stream") and transcriber._stream is not None:
                    transcriber._stream.stop()
            except Exception as e:
                print(f"Warning: Could not pause stream: {e}", file=sys.stderr)

            # Play back the response
            assistant.play_text(reply)

            # Resume audio stream
            try:
                if hasattr(transcriber, "_stream") and transcriber._stream is not None:
                    transcriber._stream.start()
            except Exception as e:
                print(f"Warning: Could not resume stream: {e}", file=sys.stderr)

            if stop_event.is_set():
                break

    except KeyboardInterrupt:
        stop_event.set()
    except Exception as e:
        print(f"Error in main loop: {e}", file=sys.stderr)
        stop_event.set()
    finally:
        transcriber.stop()
        print("Transcription stopped.", file=sys.stderr)


if __name__ == "__main__":
    main_loop()
