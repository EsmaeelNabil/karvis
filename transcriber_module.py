import mlx_whisper
import sounddevice as sd
import numpy as np
import threading
import time
import sys
import queue
import os
import contextlib
import webrtcvad  # Voice Activity Detection (Local)

# --- Constants ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE_SD = "float32"
DTYPE_PROC = "int16"

# State Management
STATE_WAITING_FOR_SPEECH = 1
STATE_RECORDING_SPEECH = 2


class RealTimeTranscriber:
    """
    Handles real-time audio streaming, VAD, and transcription, yielding results.
    """

    def __init__(
        self,
        model_name_or_path="mlx-community/whisper-tiny-mlx",
        vad_aggressiveness=3,
        vad_frame_ms=30,
        vad_silence_timeout_ms=1500,
        input_block_size=1024,
        energy_threshold=0.005,
    ):
        """
        Initializes the transcriber.

        Args:
            model_name_or_path (str): Path or HF repo name for the Whisper model.
            vad_aggressiveness (int): VAD aggressiveness (0-3).
            vad_frame_ms (int): VAD frame duration in ms (10, 20, or 30).
            vad_silence_timeout_ms (int): Silence duration to trigger transcription.
            input_block_size (int): Audio input block size.
            energy_threshold (float): Minimum audio energy to trigger transcription.
        """
        print("Initializing RealTimeTranscriber...", file=sys.stderr)
        self._model_name = model_name_or_path
        self._vad_aggressiveness = vad_aggressiveness
        self._vad_frame_ms = vad_frame_ms
        self._vad_silence_timeout_ms = vad_silence_timeout_ms
        self._input_block_size = input_block_size
        self._energy_threshold = energy_threshold

        self._vad_frame_samples = int(SAMPLE_RATE * (self._vad_frame_ms / 1000.0))
        self._max_silence_frames = int(
            self._vad_silence_timeout_ms / self._vad_frame_ms
        )

        self._vad = webrtcvad.Vad(self._vad_aggressiveness)
        print("VAD initialized.", file=sys.stderr)

        # Threading and Queues
        self._keep_running = threading.Event()  # Use Event for clearer signaling
        self._state_lock = threading.Lock()
        self._transcription_queue = queue.Queue()  # Audio chunks for worker
        self._output_queue = queue.Queue()  # Final text for generator

        # State Variables (initialized in start)
        self._current_state = STATE_WAITING_FOR_SPEECH
        self._speech_buffer = []
        self._processing_buffer = np.array([], dtype=DTYPE_PROC)
        self._silence_frames_after_speech = 0

        # Resources (initialized in start)
        self._stream = None
        self._transcriber_thread = None

    def _transcription_worker(self):
        """
        Internal thread: Waits for audio data, converts, transcribes, puts text on output queue.
        """
        while self._keep_running.is_set():
            try:
                audio_data_list = self._transcription_queue.get(timeout=0.5)
                if audio_data_list is None:  # Sentinel value to stop
                    break

                # Concatenate int16 chunks
                audio_int16 = np.concatenate(audio_data_list)
                audio_float32 = audio_int16.astype(np.float32) / 32767.0

                # Energy check
                if np.abs(audio_float32).mean() < self._energy_threshold:
                    with self._state_lock:
                        self._current_state = STATE_WAITING_FOR_SPEECH
                    continue  # Skip transcription

                # Transcribe
                text = ""
                try:
                    result = None
                    with open(os.devnull, "w") as devnull:
                        with (
                            contextlib.redirect_stdout(devnull),
                            contextlib.redirect_stderr(devnull),
                        ):
                            try:
                                result = mlx_whisper.transcribe(
                                    audio=audio_float32.flatten(),
                                    path_or_hf_repo=self._model_name,
                                    verbose=False,
                                )
                            except Exception as transcribe_exception:
                                raise transcribe_exception
                    if result:
                        text = result.get("text", "").strip()

                except Exception as e:
                    print(f"Error during transcription: {e}", file=sys.stderr)
                finally:
                    # Reset state only if transcription was attempted (passed energy check)
                    with self._state_lock:
                        self._current_state = STATE_WAITING_FOR_SPEECH

                # Put result on output queue for the generator
                if text:
                    self._output_queue.put(text)

            except queue.Empty:
                continue  # No transcription task
            except Exception as e:
                print(f"Error in transcription worker: {e}", file=sys.stderr)
                with self._state_lock:
                    self._current_state = STATE_WAITING_FOR_SPEECH
        print("Transcription worker finished.", file=sys.stderr)

    def _audio_callback(self, indata, frames, time, status):
        """Internal callback for processing audio blocks."""
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)
            return
        if not self._keep_running.is_set():
            return  # Stop processing if stop signal received

        # Convert audio to int16 for processing
        audio_int16 = (indata * 32767).astype(DTYPE_PROC).flatten()

        with self._state_lock:
            self._processing_buffer = np.concatenate(
                (self._processing_buffer, audio_int16)
            )

            while len(self._processing_buffer) >= self._vad_frame_samples:
                if not self._keep_running.is_set():
                    break  # Check flag within loop

                vad_frame = self._processing_buffer[: self._vad_frame_samples]
                self._processing_buffer = self._processing_buffer[
                    self._vad_frame_samples :
                ]

                vad_frame_bytes = vad_frame.tobytes()
                try:
                    is_speech = self._vad.is_speech(vad_frame_bytes, SAMPLE_RATE)
                except Exception as vad_err:
                    print(f"VAD error: {vad_err}", file=sys.stderr)
                    is_speech = False

                if is_speech:
                    self._silence_frames_after_speech = 0
                    if self._current_state == STATE_WAITING_FOR_SPEECH:
                        self._current_state = STATE_RECORDING_SPEECH
                        self._speech_buffer = []  # Clear buffer when speech starts
                    if self._current_state == STATE_RECORDING_SPEECH:
                        self._speech_buffer.append(vad_frame)

                elif self._current_state == STATE_RECORDING_SPEECH:
                    self._speech_buffer.append(vad_frame)  # Include trailing silence
                    self._silence_frames_after_speech += 1
                    if self._silence_frames_after_speech >= self._max_silence_frames:
                        if self._speech_buffer:
                            self._transcription_queue.put(list(self._speech_buffer))
                        self._speech_buffer = []
                        # State reset is handled by worker after transcription
                        # Break inner VAD loop for this block
                        break

    def start(self):
        """
        Starts the audio stream and transcription worker. Yields transcribed sentences.
        """
        if self._keep_running.is_set():
            print("Transcriber already running.", file=sys.stderr)
            return

        print("Starting RealTimeTranscriber...", file=sys.stderr)
        self._keep_running.set()
        self._current_state = STATE_WAITING_FOR_SPEECH
        self._speech_buffer = []
        self._processing_buffer = np.array([], dtype=DTYPE_PROC)
        self._silence_frames_after_speech = 0

        # Clear queues
        while not self._transcription_queue.empty():
            self._transcription_queue.get()
        while not self._output_queue.empty():
            self._output_queue.get()

        # Start worker thread
        self._transcriber_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self._transcriber_thread.start()

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE_SD,
                blocksize=self._input_block_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            print("Audio stream started. Listening for speech...", file=sys.stderr)

            # Generator loop: yield results from the output queue
            while self._keep_running.is_set():
                try:
                    text = self._output_queue.get(timeout=0.1)
                    if text is None:  # Check for sentinel value from stop()
                        break
                    yield text
                except queue.Empty:
                    continue  # Just check keep_running again

        except Exception as e:
            print(f"Error starting audio stream: {e}", file=sys.stderr)
            self.stop()  # Attempt to clean up if start failed
            raise  # Re-raise the exception

        print("RealTimeTranscriber start loop finished.", file=sys.stderr)

    def stop(self):
        """
        Signals the transcriber to stop streaming and processing.
        """
        if not self._keep_running.is_set():
            # print("Transcriber already stopped.", file=sys.stderr)
            return

        print("\nStopping RealTimeTranscriber...", file=sys.stderr)
        self._keep_running.clear()  # Signal threads and callback to stop

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
                print("Audio stream stopped.", file=sys.stderr)
            except Exception as sd_err:
                print(f"Error stopping sounddevice stream: {sd_err}", file=sys.stderr)
            self._stream = None

        # Signal worker thread to stop by putting sentinel value
        self._transcription_queue.put(None)

        if self._transcriber_thread is not None and self._transcriber_thread.is_alive():
            print("Waiting for transcription worker to finish...", file=sys.stderr)
            self._transcriber_thread.join(timeout=2.0)  # Wait longer for worker
            if self._transcriber_thread.is_alive():
                print(
                    "Warning: Transcription worker did not exit cleanly.",
                    file=sys.stderr,
                )
        self._transcriber_thread = None

        print("RealTimeTranscriber stopped.", file=sys.stderr)


# Example of how to potentially use this if run directly (for testing)
# if __name__ == "__main__":
#     transcriber = RealTimeTranscriber()
#     try:
#         print("Starting transcription. Press Ctrl+C to stop.")
#         for sentence in transcriber.start():
#             print(f"RECEIVED: {sentence}")
#     except KeyboardInterrupt:
#         print("Ctrl+C detected, stopping.")
#     finally:
#         transcriber.stop()
