#!/usr/bin/env python3
"""
Live real-time transcription using Moonshine ONNX + SileroVAD.
Provides a generator-based `start()` that yields final utterances.
"""

import time
import threading
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
from silero_vad import VADIterator, load_silero_vad

from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
from printer import ProgressPrinter

# --- Constants ---
SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE_SD = "float32"
CHUNK_SIZE = 512           # must be compatible with Silero VAD
LOOKBACK_CHUNKS = 5
MAX_SPEECH_SECS = 20       # force-cut after this long
MIN_REFRESH_SECS = 0.3     # interim update interval (sec)

# States
STATE_WAITING = 1
STATE_RECORDING = 2

pp = ProgressPrinter()


class RealTimeTranscriber:
    def __init__(
        self,
        model_name: str = "moonshine/base",
        vad_threshold: float = 0.5,
        vad_min_silence_ms: int = 300,
    ):
        pp.progress(f"Loading Moonshine model '{model_name}'…")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.tokenizer = load_tokenizer()

        # Warm up model to reduce first-call latency
        _ = self.model.generate(np.zeros((1, SAMPLE_RATE), dtype=np.float32))

        # Silero VAD
        self.vad = load_silero_vad(onnx=True)
        self.vad_iter = VADIterator(
            model=self.vad,
            sampling_rate=SAMPLE_RATE,
            threshold=vad_threshold,
            min_silence_duration_ms=vad_min_silence_ms,
        )

        # Buffers and queues
        self._audio_buffer = np.empty(0, dtype=np.float32)
        self._lookback = LOOKBACK_CHUNKS * CHUNK_SIZE
        self._output_queue: Queue[str] = Queue()
        self._state = STATE_WAITING

        # Thread control
        self._stop_event = threading.Event()
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        if status:
            pp.progress(f"InputStream status: {status}")
        data = indata.ravel().astype(np.float32)

        # Maintain lookback when waiting
        if self._state == STATE_WAITING:
            self._audio_buffer = np.concatenate((self._audio_buffer, data))[-self._lookback:]
        else:
            self._audio_buffer = np.concatenate((self._audio_buffer, data))

        # VAD decision
        vad_res = self.vad_iter(data)
        now = time.time()
        if vad_res:
            if "start" in vad_res and self._state == STATE_WAITING:
                self._state = STATE_RECORDING
                self._utter_start = now

            if "end" in vad_res and self._state == STATE_RECORDING:
                self._finalize_utterance()
        elif self._state == STATE_RECORDING:
            # interim updates
            if now - self._utter_start > MIN_REFRESH_SECS:
                self._utter_start = now

            # forced cut
            if len(self._audio_buffer) / SAMPLE_RATE > MAX_SPEECH_SECS:
                self._finalize_utterance()

    def _transcribe(self, audio: np.ndarray) -> str:
        tokens = self.model.generate(audio[np.newaxis, :])
        return self.tokenizer.decode_batch(tokens)[0].strip()

    def _finalize_utterance(self):
        text = self._transcribe(self._audio_buffer)
        if text:
            self._output_queue.put(text)
        # reset
        self._audio_buffer = np.empty(0, dtype=np.float32)
        self._state = STATE_WAITING

    def start(self):
        """
        Opens the mic stream and yields completed utterances as they occur.
        Use in a for-loop: `for sentence in transcriber.start(): …`
        """
        pp.progress("Starting Moonshine real-time transcriber…")
        self._stop_event.clear()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE_SD,
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        pp.progress("Audio stream live. Listening…")

        try:
            while not self._stop_event.is_set():
                try:
                    utterance = self._output_queue.get(timeout=0.1)
                    yield utterance
                except Empty:
                    continue
        finally:
            self.stop()

    def stop(self):
        """Stops audio capture and signals shutdown."""
        if self._stop_event.is_set():
            return
        pp.progress("Stopping Moonshine transcriber…")
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                pp.progress(f"Error closing stream: {e}")
            self._stream = None
        pp.progress("Transcriber stopped.")
