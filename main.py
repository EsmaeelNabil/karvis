import sys
import threading
import time

# --- Import the separated components ---
try:
    # Assuming the classes are in 'llm_tts_components.py'
    from llm_completion import OllamaCompletion
    from kokoro_tts import KokoroTTS
except ImportError:
    print("Error: Could not import OllamaCompletion or KokoroTTS.", file=sys.stderr)
    print(
        "Make sure 'llm_tts_components.py' exists and contains the classes.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Import real-time transcriber ---
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
# --- Ollama and TTS Configuration (using defaults from the classes) ---
# You can customize these parameters if needed
OLLAMA_MODEL = "cogito:14b"
KOKORO_MODEL = "prince-canuma/Kokoro-82M"
KOKORO_VOICE = "af_heart"

# --- Instantiate the handlers ---
try:
    # Instantiate Ollama completion handler
    completion_handler = OllamaCompletion(model=OLLAMA_MODEL)

    # Instantiate Kokoro TTS handler
    tts_handler = KokoroTTS(
        kokoro_model_id=KOKORO_MODEL,
        voice=KOKORO_VOICE
        # Add other KokoroTTS parameters here if needed (lang_code, speed, etc.)
    )
    # Optional: Check if TTS pipeline loaded successfully
    if tts_handler.pipeline is None:
         print("Warning: TTS Pipeline failed to load. TTS playback will be disabled.", file=sys.stderr)

except Exception as e:
    print(f"Error during initialization: {e}", file=sys.stderr)
    sys.exit(1)


# --- Transcription and Chat Loop ---
def main_loop():
    # Start real-time transcription
    transcriber = RealTimeTranscriber(
        model_name_or_path=TRANSCRIBER_MODEL, vad_silence_timeout_ms=SILENCE_TIMEOUT_MS
    )

    stop_event = threading.Event()

    # Function to listen for quit command ('q') in a separate thread
    def listener():
        print("\n*** Press 'q' then Enter to quit ***")
        while not stop_event.is_set():
            try:
                # Check if running in an interactive terminal
                if sys.stdin.isatty():
                    inp = input() # Blocks here waiting for input
                    if inp.strip().lower() == "q":
                        print("Quit command received.", file=sys.stderr)
                        stop_event.set()
                        break
                else:
                    # If not interactive (e.g., piped input), just sleep
                    # This prevents the listener from consuming CPU unnecessarily
                    # and prevents blocking if there's no stdin.
                    time.sleep(0.5)
            except EOFError: # Handle pipe closing or Ctrl+D
                 print("Input stream closed.", file=sys.stderr)
                 stop_event.set()
                 break
            except Exception as e:
                 print(f"Error in listener thread: {e}", file=sys.stderr)
                 stop_event.set()
                 break # Exit on error

    # Launch quit listener thread
    listener_thread = threading.Thread(target=listener, daemon=True)
    listener_thread.start()

    try:
        print("Starting transcription. Speak now...", file=sys.stderr)
        # Iterate through transcribed sentences
        for sentence in transcriber.start():
            if stop_event.is_set():
                print("Stop event detected, exiting transcription loop.", file=sys.stderr)
                break

            if not sentence: # Skip empty transcriptions
                continue

            print(f"\nYou: {sentence}") # Print transcribed sentence

            # 1. Generate response text using OllamaCompletion
            #    Pass only the user's sentence. The class handles the conversation structure.
            reply = completion_handler.generate_text(sentence)
            print(f"Assistant: {reply}")

            # Check again if stopped while generating text
            if stop_event.is_set():
                 print("Stop event detected after text generation.", file=sys.stderr)
                 break

            # --- Pause/Resume Audio Stream ---
            stream_paused = False
            try:
                # Check if transcriber has a controllable stream object
                if hasattr(transcriber, "_stream") and transcriber._stream is not None and hasattr(transcriber._stream, 'stop'):
                    # print("Pausing audio stream...", file=sys.stderr) # Optional debug print
                    transcriber._stream.stop()
                    stream_paused = True
                else:
                     print("Warning: Cannot access or stop transcriber stream.", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not pause stream: {e}", file=sys.stderr)
            # --- End Pause ---

            # 2. Play back the response using KokoroTTS
            #    Only attempt playback if the TTS pipeline loaded correctly
            if tts_handler.pipeline is not None and reply != "[Error generating response]":
                tts_handler.play_text(reply)
            elif reply == "[Error generating response]":
                 print("Skipping TTS playback due to generation error.", file=sys.stderr)
            else: # TTS pipeline didn't load
                 print("Skipping TTS playback as pipeline is unavailable.", file=sys.stderr)


            # Check again if stopped during TTS playback
            if stop_event.is_set():
                 print("Stop event detected after TTS playback.", file=sys.stderr)
                 # Ensure stream is resumed if it was paused before breaking
                 if stream_paused:
                    try:
                        if hasattr(transcriber, "_stream") and transcriber._stream is not None and hasattr(transcriber._stream, 'start'):
                             transcriber._stream.start()
                    except Exception as e:
                        print(f"Warning: Could not resume stream before exit: {e}", file=sys.stderr)
                 break

            # --- Resume Audio Stream ---
            if stream_paused:
                try:
                    if hasattr(transcriber, "_stream") and transcriber._stream is not None and hasattr(transcriber._stream, 'start'):
                         # print("Resuming audio stream...", file=sys.stderr) # Optional debug print
                         transcriber._stream.start()
                except Exception as e:
                    print(f"Warning: Could not resume stream: {e}", file=sys.stderr)
            # --- End Resume ---


    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping...", file=sys.stderr)
        stop_event.set()
    except Exception as e:
        print(f"\nFATAL Error in main loop: {e}", file=sys.stderr)
        stop_event.set() # Signal threads to stop
    finally:
        print("Stopping transcriber...", file=sys.stderr)
        transcriber.stop() # Ensure transcriber resources are released
        print("Transcription stopped.", file=sys.stderr)
        # Wait briefly for the listener thread to potentially notice the stop event
        if listener_thread.is_alive():
             print("Waiting for listener thread to exit...", file=sys.stderr)
             # listener_thread.join(timeout=1.0) # Optionally wait for thread


if __name__ == "__main__":
    main_loop()
    print("Main loop finished.")